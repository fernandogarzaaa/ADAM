//! ADAM MCP: JSON-RPC 2.0 stdio server exposing the organism's `adam_*`
//! tools to MCP clients (Claude Desktop, Codex, ChatGPT, etc.).

pub mod dispatch;
pub mod rpc;
pub mod tools;

pub use rpc::handle_message;
