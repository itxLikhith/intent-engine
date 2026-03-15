.PHONY: build run-crawler run-indexer run-api run-pagerank test bench fmt lint clean docker-build docker-up docker-down

# Build all binaries
build:
	@echo "Building crawler..."
	go build -o bin/crawler ./cmd/crawler
	@echo "Building indexer..."
	go build -o bin/indexer ./cmd/indexer
	@echo "Building search API..."
	go build -o bin/search-api ./cmd/search-api
	@echo "Building PageRank calculator..."
	go build -o bin/pagerank ./cmd/pagerank
	@echo "Build complete!"

# Run services
run-crawler:
	go run ./cmd/crawler

run-indexer:
	go run ./cmd/indexer

run-api:
	go run ./cmd/search-api

run-pagerank:
	go run ./cmd/pagerank

# Testing
test:
	go test -v ./...

test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

bench:
	go test -bench=. -benchmem ./...

# Code quality
fmt:
	go fmt ./...

lint:
	golangci-lint run

# Clean
clean:
	rm -rf bin/
	rm -rf data/
	go clean

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database migrations
migrate-up:
	migrate -path migrations -database "$(DATABASE_URL)" up

migrate-down:
	migrate -path migrations -database "$(DATABASE_URL)" down

# Seed URLs
seed:
	go run ./cmd/crawler -seed-only

# Help
help:
	@echo "Intent Engine Go Crawler"
	@echo ""
	@echo "Usage:"
	@echo "  make build          - Build all binaries"
	@echo "  make run-crawler    - Run crawler service"
	@echo "  make run-indexer    - Run indexer service"
	@echo "  make run-api        - Run search API"
	@echo "  make run-pagerank   - Run PageRank calculator"
	@echo "  make test           - Run tests"
	@echo "  make bench          - Run benchmarks"
	@echo "  make fmt            - Format code"
	@echo "  make lint           - Lint code"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make docker-up      - Start Docker services"
	@echo "  make docker-down    - Stop Docker services"
