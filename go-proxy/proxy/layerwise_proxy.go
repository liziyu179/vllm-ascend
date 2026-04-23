package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type LayerwiseConfig struct {
	Prefillers    []string
	Decoders      []string
	PublicBaseURL string

	MaxRetries int
	RetryDelay time.Duration

	Client *http.Client
}

type LayerwiseProxy struct {
	cfg LayerwiseConfig

	mu sync.Mutex

	prefillServers []Server
	decodeServers  []Server

	prefillSel *selector
	decodeSel  *selector

	pendingMu sync.Mutex
	pending   map[string]pendingRequest
}

type pendingRequest struct {
	APIPath     string
	RequestLen  int
	RequestData map[string]any
	OriginReqID string
}

func NewLayerwiseProxy(cfg LayerwiseConfig) (*LayerwiseProxy, error) {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.RetryDelay <= 0 {
		cfg.RetryDelay = 200 * time.Millisecond
	}
	if cfg.Client == nil {
		cfg.Client = defaultHTTPClient()
	}
	if strings.TrimSpace(cfg.PublicBaseURL) == "" {
		return nil, errors.New("public base url is required")
	}

	p := &LayerwiseProxy{
		cfg:     cfg,
		pending: make(map[string]pendingRequest),
	}
	for _, a := range cfg.Prefillers {
		s, err := newServer(a)
		if err != nil {
			return nil, fmt.Errorf("prefill addr %q: %w", a, err)
		}
		p.prefillServers = append(p.prefillServers, s)
	}
	for _, a := range cfg.Decoders {
		s, err := newServer(a)
		if err != nil {
			return nil, fmt.Errorf("decode addr %q: %w", a, err)
		}
		p.decodeServers = append(p.decodeServers, s)
	}
	if len(p.prefillServers) == 0 || len(p.decodeServers) == 0 {
		return nil, errors.New("need at least 1 prefiller and 1 decoder")
	}
	p.prefillSel = newSelector(len(p.prefillServers))
	p.decodeSel = newSelector(len(p.decodeServers))
	return p, nil
}

func (p *LayerwiseProxy) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthcheck", p.handleHealthcheck)
	mux.HandleFunc("/instances/add", p.handleInstancesAdd)
	mux.HandleFunc("/instances/remove", p.handleInstancesRemove)
	mux.HandleFunc("/v1/completions", p.handleCompletions("/completions"))
	mux.HandleFunc("/v1/chat/completions", p.handleCompletions("/chat/completions"))
	mux.HandleFunc("/v1/metaserver", p.handleMetaserver)
	return mux
}

func (p *LayerwiseProxy) handleHealthcheck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	p.mu.Lock()
	prefillN, decodeN := len(p.prefillServers), len(p.decodeServers)
	p.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":            "ok",
		"prefill_instances": prefillN,
		"decode_instances":  decodeN,
	})
}

func (p *LayerwiseProxy) handleInstancesAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req adjustReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	addrs, err := normalizeInstances(req.Instances)
	if err != nil {
		http.Error(w, "bad instances", http.StatusBadRequest)
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	switch req.Type {
	case InstancePrefill:
		added := p.addServersLocked(&p.prefillServers, p.prefillSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("add prefill instances: %v", added),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	case InstanceDecode:
		added := p.addServersLocked(&p.decodeServers, p.decodeSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("add decode instances: %v", added),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	default:
		http.Error(w, "unsupported type", http.StatusBadRequest)
	}
}

func (p *LayerwiseProxy) handleInstancesRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req adjustReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	addrs, err := normalizeInstances(req.Instances)
	if err != nil {
		http.Error(w, "bad instances", http.StatusBadRequest)
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	switch req.Type {
	case InstancePrefill:
		removed := p.removeServersLocked(&p.prefillServers, &p.prefillSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("remove prefill instances: %v", removed),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	case InstanceDecode:
		removed := p.removeServersLocked(&p.decodeServers, &p.decodeSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("remove decode instances: %v", removed),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	default:
		http.Error(w, "unsupported type", http.StatusBadRequest)
	}
}

func (p *LayerwiseProxy) addServersLocked(dst *[]Server, sel *selector, addrs []string) []string {
	exists := map[string]bool{}
	for _, s := range *dst {
		exists[s.Addr] = true
	}
	var added []string
	for _, a := range addrs {
		if exists[a] {
			continue
		}
		s, err := newServer(a)
		if err != nil {
			continue
		}
		*dst = append(*dst, s)
		added = append(added, a)
	}
	if len(added) > 0 {
		*sel = *newSelector(len(*dst))
	}
	return added
}

func (p *LayerwiseProxy) removeServersLocked(dst *[]Server, selPtr **selector, addrs []string) []string {
	toRemove := map[string]bool{}
	for _, a := range addrs {
		toRemove[a] = true
	}
	var kept []Server
	var removed []string
	for _, s := range *dst {
		if toRemove[s.Addr] {
			removed = append(removed, s.Addr)
			continue
		}
		kept = append(kept, s)
	}
	*dst = kept
	if len(*dst) > 0 {
		*selPtr = newSelector(len(*dst))
	}
	return removed
}

func (p *LayerwiseProxy) handleCompletions(apiPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body failed", http.StatusBadRequest)
			return
		}
		var reqData map[string]any
		if err := json.Unmarshal(bodyBytes, &reqData); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		requestLen := len(bodyBytes)
		requestID := newRequestID()
		apiReqID := apiRequestID(apiPath, requestID)

		p.pendingMu.Lock()
		p.pending[apiReqID] = pendingRequest{
			APIPath:     apiPath,
			RequestLen:  requestLen,
			RequestData: cloneMap(reqData),
			OriginReqID: requestID,
		}
		p.pendingMu.Unlock()

		decodeReq := cloneMap(reqData)
		decodeReq["kv_transfer_params"] = map[string]any{
			"do_remote_decode":  false,
			"do_remote_prefill": true,
			"metaserver":        strings.TrimRight(p.cfg.PublicBaseURL, "/") + "/v1/metaserver",
		}
		decodeScore := calculateDecodeScore(requestLen)

		p.mu.Lock()
		decodeIdx := p.decodeSel.acquire(decodeScore)
		decoder := p.decodeServers[decodeIdx]
		p.mu.Unlock()

		defer func() {
			p.mu.Lock()
			p.decodeSel.release(decodeIdx, decodeScore)
			p.mu.Unlock()

			p.pendingMu.Lock()
			delete(p.pending, apiReqID)
			p.pendingMu.Unlock()
		}()

		streamFlag, _ := reqData["stream"].(bool)
		if err := p.streamLayerwiseDecode(r.Context(), w, r, decoder, apiPath, decodeReq, requestID, streamFlag); err != nil {
			if !errors.Is(err, context.Canceled) {
				http.Error(w, "decode failed", http.StatusBadGateway)
			}
		}
	}
}

func (p *LayerwiseProxy) handleMetaserver(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var kvParams map[string]any
	if err := json.NewDecoder(r.Body).Decode(&kvParams); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	requestID, _ := kvParams["request_id"].(string)
	if requestID == "" {
		http.Error(w, "missing request_id", http.StatusBadRequest)
		return
	}

	p.pendingMu.Lock()
	pending, ok := p.pending[requestID]
	if ok {
		delete(p.pending, requestID)
	}
	p.pendingMu.Unlock()
	if !ok {
		http.Error(w, "unknown request_id", http.StatusNotFound)
		return
	}

	reqData := cloneMap(pending.RequestData)
	reqData["kv_transfer_params"] = kvParams
	prefillScore := calculatePrefillScore(pending.RequestLen)

	p.mu.Lock()
	prefillIdx := p.prefillSel.acquire(prefillScore)
	prefill := p.prefillServers[prefillIdx]
	p.mu.Unlock()

	defer func() {
		p.mu.Lock()
		p.prefillSel.release(prefillIdx, prefillScore)
		p.mu.Unlock()
	}()

	if _, err := p.doLayerwisePrefill(r.Context(), prefill, pending.APIPath, reqData, pending.OriginReqID); err != nil {
		http.Error(w, "prefill failed", http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (p *LayerwiseProxy) doLayerwisePrefill(
	ctx context.Context,
	server Server,
	apiPath string,
	reqData map[string]any,
	requestID string,
) (map[string]any, error) {
	prefillReq := cloneMap(reqData)
	prefillReq["stream"] = false
	prefillReq["max_tokens"] = 1
	prefillReq["min_tokens"] = 1
	if _, ok := prefillReq["max_completion_tokens"]; ok {
		prefillReq["max_completion_tokens"] = 1
	}
	delete(prefillReq, "stream_options")

	var respBytes []byte
	var err error
	for attempt := 1; attempt <= p.cfg.MaxRetries; attempt++ {
		respBytes, err = p.postJSON(ctx, server.URL+apiPath, prefillReq, requestID)
		if err == nil {
			break
		}
		if attempt < p.cfg.MaxRetries {
			select {
			case <-timeAfter(p.cfg.RetryDelay, attempt):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
	if err != nil {
		return nil, err
	}
	var resp map[string]any
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		return nil, err
	}
	if kv, ok := resp["kv_transfer_params"].(map[string]any); ok {
		return kv, nil
	}
	return nil, nil
}

func (p *LayerwiseProxy) postJSON(ctx context.Context, url string, payload any, requestID string) ([]byte, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", requestID)

	resp, err := p.cfg.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

func (p *LayerwiseProxy) streamLayerwiseDecode(
	ctx context.Context,
	w http.ResponseWriter,
	r *http.Request,
	server Server,
	apiPath string,
	reqData map[string]any,
	requestID string,
	streamFlag bool,
) error {
	originPrompt, chatFlag := extractPrompt(reqData)
	originMaxTokens := intValue(reqData["max_tokens"], 16)
	if _, ok := reqData["max_tokens"]; !ok {
		originMaxTokens = intValue(reqData["max_completion_tokens"], 16)
	}
	generated := ""
	retryCount := 0
	totalCompletionTokens := 0

	for {
		resp, err := p.doDecodeRequest(ctx, w, r, server, apiPath, reqData, requestID)
		if err != nil {
			return err
		}

		if streamFlag {
			retry, completionTokens, err := p.forwardLayerwiseStream(w, resp, &generated, retryCount > 0)
			resp.Body.Close()
			if err != nil {
				return err
			}
			if !retry {
				return nil
			}
			retryCount++
			totalCompletionTokens += completionTokens
			updatePrompt(reqData, chatFlag, originPrompt, generated)
			reqData["max_tokens"] = originMaxTokens - totalCompletionTokens + retryCount
			if _, ok := reqData["max_completion_tokens"]; ok {
				reqData["max_completion_tokens"] = originMaxTokens - totalCompletionTokens + retryCount
			}
			if maxTokens := intValue(reqData["max_tokens"], 0); maxTokens <= 0 {
				reqData["max_tokens"] = retryCount
			}
			continue
		}

		retry, completionTokens, err := p.forwardLayerwiseJSON(w, resp, &generated, retryCount > 0)
		resp.Body.Close()
		if err != nil {
			return err
		}
		if !retry {
			return nil
		}
		retryCount++
		totalCompletionTokens += completionTokens
		updatePrompt(reqData, chatFlag, originPrompt, generated)
		reqData["max_tokens"] = originMaxTokens - totalCompletionTokens + retryCount
		if _, ok := reqData["max_completion_tokens"]; ok {
			reqData["max_completion_tokens"] = originMaxTokens - totalCompletionTokens + retryCount
		}
		if maxTokens := intValue(reqData["max_tokens"], 0); maxTokens <= 0 {
			reqData["max_tokens"] = retryCount
		}
	}
}

func (p *LayerwiseProxy) doDecodeRequest(
	ctx context.Context,
	w http.ResponseWriter,
	incoming *http.Request,
	server Server,
	apiPath string,
	reqData map[string]any,
	requestID string,
) (*http.Response, error) {
	targetURL := server.URL + apiPath
	var lastErr error

	for attempt := 1; attempt <= p.cfg.MaxRetries; attempt++ {
		reqBody, err := json.Marshal(reqData)
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Request-Id", requestID)
		if auth := incoming.Header.Get("Authorization"); auth != "" {
			req.Header.Set("Authorization", auth)
		}

		resp, err := p.cfg.Client.Do(req)
		if err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			for k, vv := range resp.Header {
				for _, v := range vv {
					w.Header().Add(k, v)
				}
			}
			w.Header().Del("Content-Length")
			if stream, _ := reqData["stream"].(bool); stream {
				w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			} else {
				w.Header().Set("Content-Type", "application/json")
			}
			w.WriteHeader(http.StatusOK)
			return resp, nil
		}
		if resp != nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
		if err == nil {
			err = fmt.Errorf("decoder status %d", resp.StatusCode)
		}
		lastErr = err
		if attempt < p.cfg.MaxRetries {
			select {
			case <-timeAfter(p.cfg.RetryDelay, attempt):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
	return nil, lastErr
}

func (p *LayerwiseProxy) forwardLayerwiseStream(
	w http.ResponseWriter,
	resp *http.Response,
	generated *string,
	rewriteFinal bool,
) (retry bool, completionTokens int, err error) {
	flusher, _ := w.(http.Flusher)
	bw := bufio.NewWriterSize(w, 32*1024)
	defer bw.Flush()

	reader := bufio.NewReader(resp.Body)
	for {
		line, readErr := reader.ReadBytes('\n')
		if len(line) > 0 {
			payload, isData := parseSSEData(line)
			if isData {
				retry, token, out, perr := processLayerwisePayload(payload, generated, rewriteFinal)
				if perr != nil {
					return false, completionTokens, perr
				}
				completionTokens += token
				if retry {
					return true, completionTokens, nil
				}
				line = out
				if len(line) > 0 && !bytes.HasSuffix(line, []byte("\n")) {
					line = append([]byte("data: "), line...)
					line = append(line, '\n', '\n')
				}
			}
			if _, err := bw.Write(line); err != nil {
				return false, completionTokens, err
			}
			if flusher != nil {
				bw.Flush()
				flusher.Flush()
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return false, completionTokens, nil
			}
			return false, completionTokens, readErr
		}
	}
}

func (p *LayerwiseProxy) forwardLayerwiseJSON(
	w http.ResponseWriter,
	resp *http.Response,
	generated *string,
	rewriteFinal bool,
) (retry bool, completionTokens int, err error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, 0, err
	}
	retry, completionTokens, out, err := processLayerwisePayload(bytes.TrimSpace(body), generated, rewriteFinal)
	if err != nil {
		return false, 0, err
	}
	if retry {
		return true, completionTokens, nil
	}
	_, err = w.Write(out)
	return false, completionTokens, err
}

func processLayerwisePayload(payload []byte, generated *string, rewriteFinal bool) (bool, int, []byte, error) {
	trimmed := bytes.TrimSpace(payload)
	if bytes.Equal(trimmed, []byte("[DONE]")) || len(trimmed) == 0 {
		return false, 0, payload, nil
	}

	var chunk map[string]any
	if err := json.Unmarshal(trimmed, &chunk); err != nil {
		return false, 0, payload, nil
	}
	choices, _ := chunk["choices"].([]any)
	if len(choices) == 0 {
		return false, 0, payload, nil
	}
	choice, _ := choices[0].(map[string]any)
	content := extractChoiceContent(choice)
	*generated += content

	completionTokens := usageCompletionTokens(chunk)
	if completionTokens == 0 && content != "" {
		completionTokens = 1
	}
	if stopReason, _ := choice["stop_reason"].(string); stopReason == "recomputed" {
		return true, completionTokens, nil, nil
	}
	if rewriteFinal {
		if msg, ok := choice["message"].(map[string]any); ok {
			msg["content"] = *generated
		} else {
			choice["text"] = *generated
		}
		out, err := json.Marshal(chunk)
		if err != nil {
			return false, completionTokens, nil, err
		}
		return false, completionTokens, out, nil
	}
	return false, completionTokens, payload, nil
}

func parseSSEData(line []byte) ([]byte, bool) {
	trimmed := bytes.TrimSpace(line)
	if !bytes.HasPrefix(trimmed, []byte("data:")) {
		return nil, false
	}
	payload := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
	return payload, true
}

func extractPrompt(reqData map[string]any) (string, bool) {
	if prompt, ok := reqData["prompt"].(string); ok {
		return prompt, false
	}
	messages, ok := reqData["messages"].([]any)
	if !ok || len(messages) == 0 {
		return "", true
	}
	msg, _ := messages[0].(map[string]any)
	switch content := msg["content"].(type) {
	case string:
		return content, true
	case []any:
		if len(content) == 0 {
			return "", true
		}
		part, _ := content[0].(map[string]any)
		text, _ := part["text"].(string)
		return text, true
	default:
		return "", true
	}
}

func updatePrompt(reqData map[string]any, chat bool, originPrompt, generated string) {
	if !chat {
		reqData["prompt"] = originPrompt + generated
		return
	}
	messages, ok := reqData["messages"].([]any)
	if !ok || len(messages) == 0 {
		return
	}
	msg, _ := messages[0].(map[string]any)
	switch content := msg["content"].(type) {
	case string:
		msg["content"] = originPrompt + generated
	case []any:
		if len(content) == 0 {
			return
		}
		part, _ := content[0].(map[string]any)
		part["text"] = originPrompt + generated
	}
}

func extractChoiceContent(choice map[string]any) string {
	if delta, ok := choice["delta"].(map[string]any); ok {
		if content, ok := delta["content"].(string); ok {
			return content
		}
	}
	if msg, ok := choice["message"].(map[string]any); ok {
		if content, ok := msg["content"].(string); ok {
			return content
		}
	}
	if text, ok := choice["text"].(string); ok {
		return text
	}
	return ""
}

func usageCompletionTokens(chunk map[string]any) int {
	usage, _ := chunk["usage"].(map[string]any)
	return intValue(usage["completion_tokens"], 0)
}

func intValue(v any, def int) int {
	switch x := v.(type) {
	case int:
		return x
	case int32:
		return int(x)
	case int64:
		return int(x)
	case float64:
		return int(x)
	default:
		return def
	}
}

func apiRequestID(apiPath, requestID string) string {
	if apiPath == "/chat/completions" {
		return "chatcmpl-" + requestID
	}
	return "cmpl-" + requestID + "-0"
}

func timeAfter(base time.Duration, attempt int) <-chan time.Time {
	return time.After(base * time.Duration(1<<(attempt-1)))
}
