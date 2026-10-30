//! Turning a provider's answer about an exposed port into a core [`ExposedPortEndpoint`].
//!
//! The endpoint type, the unavailability reasons and the error builder all live in
//! `crate::sandbox::types` now and are re-exported here, so a provider module has one import for
//! everything port resolution needs. What stays is the parsing each provider does on the way:
//! a hosted sandbox reports a port as a URL (Daytona, Blaxel, Vercel) or as a bare authority
//! (E2B's `get_host`), and neither shape is something the core should know about.

use crate::sandbox::errors::SandboxResult;

pub use crate::sandbox::types::{
    ExposedPortEndpoint, ExposedPortUnavailable, exposed_port_unavailable,
};

/// Refuse a port the client never declared.
///
/// Mirrors `BaseSandboxSession._assert_exposed_port_configured`, which every provider but Blaxel
/// inherits. A provider overriding
/// [`resolve_exposed_port`](crate::sandbox::session::SandboxSession::resolve_exposed_port) calls
/// this first, because the override replaces the trait default that would otherwise do it.
pub fn assert_exposed_port_configured(
    backend: &str,
    port: u16,
    exposed_ports: &[u16],
) -> SandboxResult<()> {
    if exposed_ports.contains(&port) {
        Ok(())
    } else {
        Err(exposed_port_unavailable(
            backend,
            port,
            exposed_ports,
            ExposedPortUnavailable::NotConfigured,
        ))
    }
}

/// Split an authority into its host and its explicit port, if it carries one.
///
/// A bracketed IPv6 literal keeps its colons; `None` means the authority has no usable host.
fn split_authority(authority: &str) -> Option<(String, Option<u16>)> {
    let authority = match authority.rsplit_once('@') {
        Some((_userinfo, rest)) => rest,
        None => authority,
    };
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, tail) = rest.split_once(']')?;
        if host.is_empty() {
            return None;
        }
        let port = match tail.strip_prefix(':') {
            Some(port) => Some(port.parse().ok()?),
            None if tail.is_empty() => None,
            None => return None,
        };
        return Some((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        // A second colon means a bare IPv6 literal, which carries no port.
        Some((host, _)) if host.contains(':') => Some((authority.to_string(), None)),
        Some((host, port)) if !host.is_empty() => {
            Some((host.to_string(), Some(port.parse().ok()?)))
        }
        Some(_) => None,
        None => Some((authority.to_string(), None)),
    }
}

/// Parse a provider-issued URL into an endpoint.
///
/// Returns `None` when the URL carries no host. A URL without an explicit port falls back to the
/// scheme default, as every Python provider does.
pub fn endpoint_from_url(url: &str) -> Option<ExposedPortEndpoint> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest
        .find(['/', '?', '#'])
        .map(|index| &rest[..index])
        .unwrap_or(rest);
    let (host, port) = split_authority(authority)?;
    let tls = matches!(scheme.to_ascii_lowercase().as_str(), "https" | "wss");
    Some(ExposedPortEndpoint::new(
        host,
        port.unwrap_or(if tls { 443 } else { 80 }),
        tls,
    ))
}

/// Parse a bare `host` or `host:port` authority, which is what E2B's `get_host` returns.
///
/// An authority with an explicit port is a plaintext endpoint; a bare host is served over TLS on
/// 443. Mirrors `_e2b_endpoint_from_host`.
pub fn endpoint_from_authority(authority: &str) -> Option<ExposedPortEndpoint> {
    let (host, port) = split_authority(authority)?;
    match port {
        Some(port) => Some(ExposedPortEndpoint::new(host, port, false)),
        None => Some(ExposedPortEndpoint::new(host, 443, true)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::errors::{ErrorCode, SandboxOp};

    #[test]
    fn reads_an_authority_the_way_e2b_reports_it() {
        assert_eq!(
            endpoint_from_authority("3000-abc.e2b.app"),
            Some(ExposedPortEndpoint::new("3000-abc.e2b.app", 443, true))
        );
        assert_eq!(
            endpoint_from_authority("127.0.0.1:49160"),
            Some(ExposedPortEndpoint::new("127.0.0.1", 49160, false))
        );
        assert_eq!(endpoint_from_authority(""), None);
    }

    #[test]
    fn reads_a_provider_preview_url() {
        assert_eq!(
            endpoint_from_url("https://preview.example.com/app"),
            Some(ExposedPortEndpoint::new("preview.example.com", 443, true))
        );
        assert_eq!(
            endpoint_from_url("http://preview.example.com:8080/"),
            Some(ExposedPortEndpoint::new("preview.example.com", 8080, false))
        );
        assert_eq!(endpoint_from_url("not a url"), None);
        assert_eq!(
            endpoint_from_url("ws://[::1]:9000/socket"),
            Some(ExposedPortEndpoint::new("::1", 9000, false))
        );
    }

    #[test]
    fn refuses_a_port_the_client_never_declared() {
        let error = assert_exposed_port_configured("e2b", 3000, &[8080]).unwrap_err();
        assert_eq!(error.code, ErrorCode::ExposedPortUnavailable);
        assert_eq!(error.op, Some(SandboxOp::ResolveExposedPort));
        assert_eq!(
            error.context.get("reason").map(String::as_str),
            Some("not_configured")
        );
        assert_eq!(error.retryable, Some(false));
        assert!(assert_exposed_port_configured("e2b", 3000, &[3000]).is_ok());
    }
}
