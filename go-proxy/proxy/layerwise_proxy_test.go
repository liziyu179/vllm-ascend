package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func newLayerwisePrefillerHandler(t *testing.T, c *backendCounters) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != "/v1/completions" && r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		c.prefillCalls.Add(1)

		var m map[string]any
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if v, _ := m["stream"].(bool); v {
			http.Error(w, "prefill must be non-stream", http.StatusBadRequest)
			return
		}
		if mt := intValue(m["max_tokens"], -1); mt != 1 {
			http.Error(w, "max_tokens must be 1", http.StatusBadRequest)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"kv_transfer_params": map[string]any{
				"prefill_done": true,
			},
		})
	})
}

func newLayerwiseDecoderHandler(t *testing.T, c *backendCounters, client *http.Client) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != "/v1/completions" && r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		c.decodeCalls.Add(1)

		var m map[string]any
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		kv, _ := m["kv_transfer_params"].(map[string]any)
		metaserver, _ := kv["metaserver"].(string)
		if metaserver == "" {
			http.Error(w, "missing metaserver", http.StatusBadRequest)
			return
		}

		apiPath := strings.TrimPrefix(r.URL.Path, "/v1")
		requestID := apiRequestID(apiPath, r.Header.Get("X-Request-Id"))
		payload := map[string]any{
			"request_id": requestID,
			"connector":  "mock-layerwise",
		}
		b, _ := json.Marshal(payload)
		req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, metaserver, bytes.NewReader(b))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			http.Error(w, fmt.Sprintf("metaserver status %d", resp.StatusCode), http.StatusBadGateway)
			return
		}

		streamFlag, _ := m["stream"].(bool)
		if streamFlag {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			fl, _ := w.(http.Flusher)
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"))
			if fl != nil {
				fl.Flush()
			}
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			if fl != nil {
				fl.Flush()
			}
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"id":      requestID,
			"choices": []any{map[string]any{"text": "ok"}},
			"usage":   map[string]any{"completion_tokens": 1},
		})
	})
}

func TestLayerwiseProxy_10000Concurrent(t *testing.T) {
	t.Parallel()

	var b0, b1 backendCounters

	tr := newInmemTransport()
	client := &http.Client{Transport: tr}

	tr.Register("prefill0:1", newLayerwisePrefillerHandler(t, &b0))
	tr.Register("prefill1:1", newLayerwisePrefillerHandler(t, &b1))
	tr.Register("decode0:1", newLayerwiseDecoderHandler(t, &b0, client))
	tr.Register("decode1:1", newLayerwiseDecoderHandler(t, &b1, client))

	p, err := NewLayerwiseProxy(LayerwiseConfig{
		Prefillers:    []string{"prefill0:1", "prefill1:1"},
		Decoders:      []string{"decode0:1", "decode1:1"},
		PublicBaseURL: "http://proxy.local",
		MaxRetries:    3,
		RetryDelay:    1 * time.Millisecond,
		Client:        client,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := p.Handler()
	tr.Register("proxy.local", h)

	const N = 10000
	var okCount atomic.Int64
	var status4xx atomic.Int64
	var status5xx atomic.Int64
	var wg sync.WaitGroup
	wg.Add(N)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	firstErr := make(chan string, 20)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			reqBody := map[string]any{
				"model":      "m",
				"prompt":     fmt.Sprintf("p-%d", i),
				"max_tokens": 16,
				"stream":     false,
			}
			b, _ := json.Marshal(reqBody)
			req := httptest.NewRequest(http.MethodPost, "http://proxy.local/v1/completions", bytes.NewReader(b)).WithContext(ctx)
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				if rr.Code >= 400 && rr.Code < 500 {
					status4xx.Add(1)
				} else if rr.Code >= 500 {
					status5xx.Add(1)
				}
				select {
				case firstErr <- fmt.Sprintf("status %d body=%q", rr.Code, rr.Body.String()):
				default:
				}
				return
			}

			var out map[string]any
			if err := json.NewDecoder(rr.Body).Decode(&out); err != nil {
				select {
				case firstErr <- fmt.Sprintf("decode err: %v", err):
				default:
				}
				return
			}
			choices, _ := out["choices"].([]any)
			if len(choices) == 0 {
				select {
				case firstErr <- "empty choices":
				default:
				}
				return
			}
			okCount.Add(1)
		}(i)
	}

	wg.Wait()
	if got := okCount.Load(); got != N {
		var samples []string
		for {
			select {
			case s := <-firstErr:
				samples = append(samples, s)
			default:
				goto done
			}
		}
	done:
		t.Fatalf("ok %d/%d (4xx=%d 5xx=%d) samples=%v", got, N, status4xx.Load(), status5xx.Load(), samples)
	}

	p0c := b0.prefillCalls.Load()
	p1c := b1.prefillCalls.Load()
	d0c := b0.decodeCalls.Load()
	d1c := b1.decodeCalls.Load()
	if p0c == 0 || p1c == 0 || d0c == 0 || d1c == 0 {
		t.Fatalf("uneven distribution: prefill(%d,%d) decode(%d,%d)", p0c, p1c, d0c, d1c)
	}
}

func TestLayerwiseProxy_Stream(t *testing.T) {
	t.Parallel()

	var b backendCounters

	tr := newInmemTransport()
	client := &http.Client{Transport: tr}
	tr.Register("prefill0:1", newLayerwisePrefillerHandler(t, &b))
	tr.Register("decode0:1", newLayerwiseDecoderHandler(t, &b, client))

	p, err := NewLayerwiseProxy(LayerwiseConfig{
		Prefillers:    []string{"prefill0:1"},
		Decoders:      []string{"decode0:1"},
		PublicBaseURL: "http://proxy.local",
		MaxRetries:    1,
		RetryDelay:    1 * time.Millisecond,
		Client:        client,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := p.Handler()
	tr.Register("proxy.local", h)

	reqBody := map[string]any{
		"model":      "m",
		"prompt":     "hello",
		"max_tokens": 16,
		"stream":     true,
	}
	bb, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "http://proxy.local/v1/completions", bytes.NewReader(bb))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rr.Code, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}
	if body := rr.Body.String(); !strings.Contains(body, "[DONE]") {
		t.Fatalf("missing DONE: %q", body)
	}
}
