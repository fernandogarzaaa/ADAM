FROM rust:1-bookworm AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p adam-mcp

FROM debian:bookworm-slim
RUN useradd --system --create-home adam
COPY --from=builder /build/target/release/adam-mcp /usr/local/bin/adam-mcp
USER adam
WORKDIR /home/adam
ENV ADAM_MEMORY_PATH=/home/adam/adam_memory.db
ENTRYPOINT ["adam-mcp"]
