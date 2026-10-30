//! The `rmcp` client handler behind [`crate::mcp::MCPServerConfig::message_handler`].
//!
//! Python hands every session message to `message_handler`. `rmcp` dispatches each server
//! notification to its own hook instead, so [`MCPClientHandler`] re-assembles the JSON-RPC
//! shape (`{"method": …, "params": …}`) and forwards it to one callback.

use rmcp::RoleClient;
#[allow(deprecated)]
use rmcp::model::LoggingMessageNotificationParam;
use rmcp::model::{
    CancelledNotificationParam, CustomNotification, ProgressNotificationParam,
    ResourceUpdatedNotificationParam, SubscriptionsAcknowledgedNotificationParams,
    TaskStatusNotificationParams,
};
use rmcp::service::NotificationContext;
use serde::Serialize;
use serde_json::{Value, json};

use super::MCPMessageHandler;

/// Forwards `rmcp` client notifications to the server's `message_handler`.
#[derive(Clone, Default)]
pub(crate) struct MCPClientHandler {
    message_handler: Option<MCPMessageHandler>,
}

impl MCPClientHandler {
    pub(crate) fn new(message_handler: Option<MCPMessageHandler>) -> Self {
        MCPClientHandler { message_handler }
    }

    fn emit(&self, method: &str, params: Option<Value>) {
        if let Some(handler) = &self.message_handler {
            let mut message = json!({"jsonrpc": "2.0", "method": method});
            if let Some(params) = params {
                message["params"] = params;
            }
            handler(&message);
        }
    }

    fn emit_with<P: Serialize>(&self, method: &str, params: P) {
        if self.message_handler.is_some() {
            self.emit(method, serde_json::to_value(params).ok());
        }
    }
}

impl rmcp::handler::client::ClientHandler for MCPClientHandler {
    fn on_cancelled(
        &self,
        params: CancelledNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/cancelled", params);
        std::future::ready(())
    }

    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/progress", params);
        std::future::ready(())
    }

    #[allow(deprecated)]
    fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/message", params);
        std::future::ready(())
    }

    fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/resources/updated", params);
        std::future::ready(())
    }

    fn on_resource_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit("notifications/resources/list_changed", None);
        std::future::ready(())
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit("notifications/tools/list_changed", None);
        std::future::ready(())
    }

    fn on_prompt_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit("notifications/prompts/list_changed", None);
        std::future::ready(())
    }

    fn on_subscriptions_acknowledged(
        &self,
        params: SubscriptionsAcknowledgedNotificationParams,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/subscriptions/acknowledged", params);
        std::future::ready(())
    }

    fn on_task_status(
        &self,
        params: TaskStatusNotificationParams,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.emit_with("notifications/tasks/status", params);
        std::future::ready(())
    }

    fn on_custom_notification(
        &self,
        notification: CustomNotification,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + Send + '_ {
        if self.message_handler.is_some() {
            match serde_json::to_value(&notification) {
                Ok(value) => self.emit(
                    value
                        .get("method")
                        .and_then(Value::as_str)
                        .unwrap_or("notifications/custom"),
                    value.get("params").cloned(),
                ),
                Err(e) => log::warn!("Ignoring unserializable MCP notification: {e}"),
            }
        }
        std::future::ready(())
    }
}
