package crawler

import "time"

// Config holds crawler configuration
type Config struct {
	// Concurrency
	MaxConcurrentRequests int           `yaml:"max_concurrent_requests"`
	MaxConnectionsPerHost int           `yaml:"max_connections_per_host"`

	// Limits
	MaxPages          int `yaml:"max_pages"`
	MaxDepth          int `yaml:"max_depth"`
	MaxPagesPerDomain int `yaml:"max_pages_per_domain"`

	// Timing
	RequestTimeout    time.Duration `yaml:"request_timeout"`
	CrawlDelay        time.Duration `yaml:"crawl_delay"`
	RandomDelay       time.Duration `yaml:"random_delay"`
	RespectRobotsTxt  bool          `yaml:"respect_robots_txt"`
	UserAgent         string        `yaml:"user_agent"`
	SkipTLSVerification bool        `yaml:"skip_tls_verification"`

	// Content
	MaxContentSize      int      `yaml:"max_content_size"`
	AllowedContentTypes []string `yaml:"allowed_content_types"`

	// Domains
	AllowedDomains  []string `yaml:"allowed_domains"`
	BlockedDomains  []string `yaml:"blocked_domains"`

	// URL patterns to skip
	SkipURLPatterns []string `yaml:"skip_url_patterns"`

	// Redis
	RedisAddr     string `yaml:"redis_addr"`
	RedisPassword string `yaml:"redis_password"`
	RedisDB       int    `yaml:"redis_db"`

	// Storage
	BadgerPath   string `yaml:"badger_path"`
	PostgresDSN  string `yaml:"postgres_dsn"`
	BlevePath    string `yaml:"bleve_path"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		// Concurrency
		MaxConcurrentRequests: 10,
		MaxConnectionsPerHost: 2,

		// Limits
		MaxPages:          10000,
		MaxDepth:          5,
		MaxPagesPerDomain: 1000,

		// Timing
		RequestTimeout:    30 * time.Second,
		CrawlDelay:        1 * time.Second,
		RandomDelay:       500 * time.Millisecond,
		RespectRobotsTxt:  true,
		UserAgent:         "IntentEngineBot/1.0 (+https://github.com/itxLikhith/intent-engine)",
		SkipTLSVerification: true, // For development, skip TLS verification

		// Content
		MaxContentSize: 10 * 1024 * 1024, // 10MB
		AllowedContentTypes: []string{
			"text/html",
			"application/xhtml+xml",
		},

		// Domains
		AllowedDomains: []string{}, // Empty = all domains
		BlockedDomains: []string{
			"facebook.com",
			"twitter.com",
			"instagram.com",
			"linkedin.com",
		},

		// URL patterns to skip
		SkipURLPatterns: []string{
			`\.(jpg|jpeg|png|gif|svg|ico|webp)$`,
			`\.(pdf|doc|docx|xls|xlsx|ppt|pptx)$`,
			`\.(zip|rar|tar|gz|7z)$`,
			`\.(mp3|mp4|avi|mov|wmv)$`,
			`\.(css|js|woff|woff2|ttf|eot)$`,
			`\/(login|signup|register|checkout|cart)`,
			`\?.*(session|token|auth)=`,
		},

		// Redis
		RedisAddr:     "localhost:6379",
		RedisPassword: "",
		RedisDB:       0,

		// Storage
		BadgerPath:  "./data/badger",
		PostgresDSN: "postgresql://user:pass@localhost:5432/intent_engine",
		BlevePath:   "./data/bleve",
	}
}
