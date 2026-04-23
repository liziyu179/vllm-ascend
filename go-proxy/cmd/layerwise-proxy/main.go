package main

import (
	"flag"
	"log"
	"net/http"
	"strings"
	"time"

	"example.com/loadbalance-proxy/proxy"
)

func main() {
	var (
		listenAddr    = flag.String("listen", "127.0.0.1:9001", "proxy listen addr")
		publicBaseURL = flag.String("public-base-url", "http://127.0.0.1:9001", "public base url reachable by decoders")
		prefillers    = flag.String("prefillers", "127.0.0.1:8100", "comma-separated prefill addrs host:port")
		decoders      = flag.String("decoders", "127.0.0.1:8200", "comma-separated decode addrs host:port")
		maxRetries    = flag.Int("max-retries", 3, "max retries for backend requests")
		retryDelay    = flag.Duration("retry-delay", 200*time.Millisecond, "base retry delay")
	)
	flag.Parse()

	p, err := proxy.NewLayerwiseProxy(proxy.LayerwiseConfig{
		Prefillers:    splitCSV(*prefillers),
		Decoders:      splitCSV(*decoders),
		PublicBaseURL: *publicBaseURL,
		MaxRetries:    *maxRetries,
		RetryDelay:    *retryDelay,
	})
	if err != nil {
		log.Fatal(err)
	}

	srv := &http.Server{
		Addr:         *listenAddr,
		Handler:      p.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0,
		IdleTimeout:  90 * time.Second,
	}
	log.Printf("layerwise proxy listening on %s", *listenAddr)
	log.Fatal(srv.ListenAndServe())
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
