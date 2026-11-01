//! Procedural macros for `openai-agents-sdk`. Use them through the `macros` feature of the main
//! crate: `openai_agents_sdk::function_tool`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Expr, ExprLit, FnArg, GenericArgument, ImplItemFn, ItemFn, Lit, Meta, Pat,
    PathArguments, ReturnType, Token, Type, parse_macro_input,
};

/// Options of `#[function_tool(...)]`.
#[derive(Default)]
struct ToolOptions {
    name: Option<String>,
    description: Option<String>,
    strict: Option<bool>,
    /// Marks an associated function (no `self`) inside an `impl` block, which the attribute
    /// cannot tell apart from a free function.
    associated: bool,
}

impl Parse for ToolOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut opts = ToolOptions::default();
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        for meta in metas {
            if let Meta::Path(path) = &meta
                && path.is_ident("associated")
            {
                opts.associated = true;
                continue;
            }
            let Meta::NameValue(nv) = &meta else {
                return Err(syn::Error::new_spanned(meta, "expected `key = value`"));
            };
            let key = nv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            match (key.as_str(), &nv.value) {
                (
                    "name",
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }),
                ) => opts.name = Some(s.value()),
                (
                    "description",
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }),
                ) => opts.description = Some(s.value()),
                (
                    "strict",
                    Expr::Lit(ExprLit {
                        lit: Lit::Bool(b), ..
                    }),
                ) => opts.strict = Some(b.value),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &nv.path,
                        "unknown option; expected `name`, `description` or `strict`",
                    ));
                }
            }
        }
        Ok(opts)
    }
}

fn doc_lines(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .map(|l| l.strip_prefix(' ').unwrap_or(&l).to_string())
        .collect()
}

/// Split a doc comment into the description and per-argument descriptions. Recognizes
/// rustdoc (`# Arguments` with `* \`name\` - text`), Google (`Args:` with `name: text`), NumPy
/// (`Parameters` over a dashed rule) and Sphinx (`:param name: text`) sections. The description is
/// the text before the first section, so `Returns` and the like never reach it.
fn parse_docs(lines: &[String]) -> (String, Vec<(String, String)>) {
    if lines.iter().any(|l| l.trim_start().starts_with(":param ")) {
        return parse_sphinx_docs(lines);
    }
    if let Some(at) = lines.windows(2).position(|w| is_numpy_header(&w[0], &w[1])) {
        return parse_numpy_docs(lines, at);
    }
    let mut description = Vec::new();
    let mut args: Vec<(String, String)> = Vec::new();
    let mut in_args = false;
    let mut in_description = true;
    for line in lines {
        let t = line.trim();
        let lower = t.to_ascii_lowercase();
        if lower == "# arguments" || lower == "# args" || lower == "args:" || lower == "arguments:"
        {
            in_args = true;
            in_description = false;
            continue;
        }
        if in_args {
            if t.starts_with('#') || (t.ends_with(':') && !t.contains(' ')) {
                in_args = false;
                continue;
            }
            let item = t
                .trim_start_matches(['*', '-'])
                .trim()
                .trim_start_matches('`');
            let split = item
                .split_once("` -")
                .or_else(|| item.split_once("`:"))
                .or_else(|| item.split_once(" - "))
                .or_else(|| item.split_once(':'));
            match split {
                Some((name, text)) if !name.trim().contains(' ') && !name.trim().is_empty() => {
                    args.push((
                        name.trim().trim_matches('`').to_string(),
                        text.trim().to_string(),
                    ));
                }
                _ if !t.is_empty() => {
                    // A wrapped description keeps its line break, as griffe does.
                    if let Some(last) = args.last_mut() {
                        last.1.push('\n');
                        last.1.push_str(t);
                    }
                }
                _ => {}
            }
            continue;
        }
        if t.starts_with('#') || (t.ends_with(':') && !t.contains(' ') && t.len() > 1) {
            // Any other section (`Returns:`, `Note:`, `# Examples`) ends the description.
            in_description = false;
            continue;
        }
        if in_description {
            description.push(line.clone());
        }
    }
    let description = description.join("\n").trim().to_string();
    (description, args)
}

/// A NumPy section title: a word over a rule of dashes.
fn is_numpy_header(title: &str, rule: &str) -> bool {
    let rule = rule.trim();
    !title.trim().is_empty()
        && !rule.is_empty()
        && rule.chars().all(|c| c == '-')
        && !title.starts_with(char::is_whitespace)
}

/// NumPy style: `Parameters` over dashes, `name: type` entries with an indented description.
fn parse_numpy_docs(lines: &[String], header: usize) -> (String, Vec<(String, String)>) {
    let description = lines[..header].join("\n").trim().to_string();
    let mut args: Vec<(String, String)> = Vec::new();
    if !lines[header].trim().eq_ignore_ascii_case("parameters") {
        return (description, args);
    }
    let mut i = header + 2;
    while i < lines.len() {
        let line = &lines[i];
        if lines
            .get(i + 1)
            .is_some_and(|next| is_numpy_header(line, next))
        {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            i += 1;
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            if let Some(last) = args.last_mut() {
                if !last.1.is_empty() {
                    last.1.push('\n');
                }
                last.1.push_str(t);
            }
        } else {
            let name = t.split(':').next().unwrap_or(t).trim();
            args.push((name.to_string(), String::new()));
        }
        i += 1;
    }
    (description, args)
}

/// Sphinx style: `:param name: text` fields, other `:field:` lines ignored.
fn parse_sphinx_docs(lines: &[String]) -> (String, Vec<(String, String)>) {
    let mut description = Vec::new();
    let mut args: Vec<(String, String)> = Vec::new();
    let mut in_fields = false;
    let mut continuing = false;
    for line in lines {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(":param ") {
            in_fields = true;
            // `:param type name: text` keeps the last word before the colon as the name.
            if let Some((head, text)) = rest.split_once(':')
                && let Some(name) = head.split_whitespace().last()
            {
                args.push((name.to_string(), text.trim().to_string()));
                continuing = true;
                continue;
            }
            continuing = false;
        } else if t.starts_with(':') {
            in_fields = true;
            continuing = false;
        } else if in_fields {
            if t.is_empty() {
                continuing = false;
            } else if continuing && let Some(last) = args.last_mut() {
                last.1.push('\n');
                last.1.push_str(t);
            }
        } else {
            description.push(line.clone());
        }
    }
    (description.join("\n").trim().to_string(), args)
}

/// `ToolContext<C>` -> `Some(C)`.
fn tool_context_type(ty: &Type) -> Option<Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "ToolContext" {
        return None;
    }
    match &seg.arguments {
        PathArguments::AngleBracketed(a) => a.args.iter().find_map(|g| match g {
            GenericArgument::Type(t) => Some(t.clone()),
            _ => None,
        }),
        _ => None,
    }
}

fn returns_result(ret: &ReturnType) -> bool {
    match ret {
        ReturnType::Type(_, ty) => match ty.as_ref() {
            Type::Path(p) => p.path.segments.last().is_some_and(|s| s.ident == "Result"),
            _ => false,
        },
        ReturnType::Default => false,
    }
}

fn pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Turn a function into an agent tool.
///
/// The function keeps working as written, and a constructor named `<fn>_tool()` is generated
/// that returns a `FunctionTool`. The tool name is the function name, the description and
/// parameter descriptions come from the doc comment (`# Arguments` or `Args:` sections, or
/// `#[arg(desc = "...")]` on a parameter), and the parameter schema is derived with
/// `schemars`.
///
/// An optional first parameter of type `ToolContext<C>` receives the tool context and fixes the
/// context type; otherwise the constructor is generic over `C`. The function may be `async` or
/// not and may return any `IntoToolOutput` value or a `Result` whose error converts into
/// `AgentsError`.
///
/// Options: `#[function_tool(name = "...", description = "...", strict = false)]`.
///
/// Inside an `impl` block, a function with a `self` receiver gets `<fn>_tool(&self)`, which binds
/// a clone of the receiver (the type must be `Clone + Send + Sync + 'static`; `&mut self` is not
/// supported). An associated function without a receiver is marked
/// `#[function_tool(associated)]` and gets `<fn>_tool()`; the attribute cannot see that it sits
/// in an `impl` block. The parameter struct is local to the constructor.
#[proc_macro_attribute]
pub fn function_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let opts = parse_macro_input!(attr as ToolOptions);
    let tokens = TokenStream2::from(item);
    // A free function, or a function inside an `impl` block: one with a `self` receiver, or
    // marked `associated`, is the latter (the two read the same to an attribute macro).
    let in_impl = opts.associated
        || syn::parse2::<ImplItemFn>(tokens.clone()).is_ok_and(|f| f.sig.receiver().is_some());
    let expanded = if in_impl {
        syn::parse2::<ImplItemFn>(tokens).and_then(|mut method| expand_in_impl(opts, &mut method))
    } else {
        syn::parse2::<ItemFn>(tokens).and_then(|mut func| expand(opts, &mut func))
    };
    match expanded {
        Ok(t) => t.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// What one function parameter list contributes to a tool.
struct Analysis {
    tool_name: String,
    description: String,
    context_type: Option<Type>,
    fields: Vec<TokenStream2>,
    call_args: Vec<TokenStream2>,
    /// A `self` receiver, which only a function inside an `impl` block has.
    receiver: bool,
}

fn analyze(
    opts: ToolOptions,
    sig: &mut syn::Signature,
    attrs: &[Attribute],
    allow_receiver: bool,
) -> syn::Result<Analysis> {
    let fn_ident = sig.ident.clone();
    let (doc_description, doc_args) = parse_docs(&doc_lines(attrs));
    let tool_name = opts.name.unwrap_or_else(|| fn_ident.to_string());
    let description = opts.description.unwrap_or(doc_description);

    let mut context_type: Option<Type> = None;
    let mut fields = Vec::new();
    let mut call_args = Vec::new();
    let mut receiver = false;
    let mut typed_index = 0usize;
    for input in sig.inputs.iter_mut() {
        let pt = match input {
            FnArg::Receiver(r) if allow_receiver => {
                if r.reference.is_some() && r.mutability.is_some() {
                    return Err(syn::Error::new_spanned(
                        r,
                        "#[function_tool] cannot bind a `&mut self` receiver; use `&self` and \
                         interior mutability",
                    ));
                }
                receiver = true;
                continue;
            }
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[function_tool] on a free function cannot take `self`",
                ));
            }
            FnArg::Typed(pt) => pt,
        };
        let i = typed_index;
        typed_index += 1;
        if i == 0
            && let Some(c) = tool_context_type(&pt.ty)
        {
            context_type = Some(c);
            call_args.push(quote!(__ctx));
            continue;
        }
        let Pat::Ident(pi) = pt.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &pt.pat,
                "tool parameters must be plain identifiers",
            ));
        };
        let ident = pi.ident.clone();
        let ty = pt.ty.as_ref().clone();
        // `#[arg(desc = "...")]` overrides the doc comment and is removed from the function.
        let mut desc: Option<String> = None;
        let mut kept = Vec::new();
        for a in pt.attrs.drain(..) {
            if a.path().is_ident("arg") {
                a.parse_nested_meta(|m| {
                    if m.path.is_ident("desc") || m.path.is_ident("description") {
                        let s: syn::LitStr = m.value()?.parse()?;
                        desc = Some(s.value());
                        Ok(())
                    } else {
                        Err(m.error("expected `desc`"))
                    }
                })?;
            } else {
                kept.push(a);
            }
        }
        pt.attrs = kept;
        let name = ident.to_string();
        let name = name.trim_start_matches("r#").to_string();
        let desc = desc.or_else(|| {
            doc_args
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, d)| d.clone())
        });
        let doc = desc.map(|d| quote!(#[doc = #d]));
        fields.push(quote! { #doc pub #ident: #ty });
        call_args.push(quote!(__args.#ident));
    }
    Ok(Analysis {
        tool_name,
        description,
        context_type,
        fields,
        call_args,
        receiver,
    })
}

/// A function inside an `impl` block. The function keeps working as written, and a sibling
/// constructor `<fn>_tool` is generated in the same block: for an associated function it takes
/// no arguments; for a method it takes `&self` and calls the method on a clone of the receiver,
/// so the type must be `Clone + Send + Sync + 'static`. The parameter struct is local to the
/// constructor, so it cannot name the impl's generic parameters.
fn expand_in_impl(opts: ToolOptions, func: &mut ImplItemFn) -> syn::Result<TokenStream2> {
    let sdk = quote!(::openai_agents_sdk);
    let fn_ident = func.sig.ident.clone();
    let vis = func.vis.clone();
    let strict = opts.strict;
    let a = analyze(opts, &mut func.sig, &func.attrs, true)?;
    let args_ident = format_ident!("__{}Args", pascal_case(&fn_ident.to_string()));
    let ctor_ident = format_ident!("{}_tool", fn_ident);
    let (tool_name, description) = (&a.tool_name, &a.description);
    let fields = &a.fields;
    let call_args = &a.call_args;
    let callee = if a.receiver {
        quote!(__this.#fn_ident)
    } else {
        quote!(Self::#fn_ident)
    };
    let is_async = func.sig.asyncness.is_some();
    let call = if is_async {
        quote!(#callee(#(#call_args),*).await)
    } else if a.receiver {
        quote!(#sdk::__private::run_blocking(move || __this.#fn_ident(#(#call_args),*)).await)
    } else {
        quote!(#sdk::__private::run_blocking(move || Self::#fn_ident(#(#call_args),*)).await)
    };
    let result = if returns_result(&func.sig.output) {
        quote!(#call.map_err(::core::convert::Into::<#sdk::AgentsError>::into))
    } else {
        quote!(::core::result::Result::<_, #sdk::AgentsError>::Ok(#call))
    };
    let ctx_pat = if a.context_type.is_some() {
        quote!(__ctx)
    } else {
        quote!(_)
    };
    let (generics, ctx_ty, ctx_bound) = match &a.context_type {
        Some(c) => (quote!(), quote!(#c), quote!()),
        None => (
            quote!(<__C>),
            quote!(__C),
            quote!(__C: ::core::marker::Send + ::core::marker::Sync + 'static,),
        ),
    };
    let receiver_param = if a.receiver { quote!(&self) } else { quote!() };
    let receiver_bound = if a.receiver {
        quote!(Self: ::core::clone::Clone + ::core::marker::Send + ::core::marker::Sync + 'static,)
    } else {
        quote!(Self: 'static,)
    };
    let bind = if a.receiver {
        quote!(let __this = ::core::clone::Clone::clone(self);)
    } else {
        quote!()
    };
    let per_call = if a.receiver {
        quote!(let __this = ::core::clone::Clone::clone(&__this);)
    } else {
        quote!()
    };
    let strict = strict
        .map(|s| quote!(.with_strict_json_schema(#s)))
        .unwrap_or_default();
    let ctor_doc = format!("Build the `{tool_name}` function tool from [`Self::{fn_ident}`].");
    let schemars_path = format!("{sdk}::__private::schemars").replace(' ', "");
    let serde_path = format!("{sdk}::__private::serde").replace(' ', "");
    Ok(quote! {
        #func

        #[doc = #ctor_doc]
        #vis fn #ctor_ident #generics (#receiver_param) -> #sdk::FunctionTool<#ctx_ty>
        where #ctx_bound #receiver_bound
        {
            #[allow(non_camel_case_types, dead_code)]
            #[derive(#sdk::__private::serde::Deserialize, #sdk::__private::schemars::JsonSchema)]
            #[serde(crate = #serde_path)]
            #[schemars(crate = #schemars_path)]
            struct #args_ident {
                #(#fields,)*
            }
            #bind
            #sdk::FunctionTool::<#ctx_ty>::new(
                #tool_name,
                #description,
                move |#ctx_pat: #sdk::tool_context::ToolContext<#ctx_ty>, __args: #args_ident| {
                    #per_call
                    async move {
                        #result
                    }
                },
            )
            #strict
        }
    })
}

fn expand(opts: ToolOptions, func: &mut ItemFn) -> syn::Result<TokenStream2> {
    let sdk = quote!(::openai_agents_sdk);
    let fn_ident = func.sig.ident.clone();
    let vis = func.vis.clone();
    let strict_opt = opts.strict;
    let Analysis {
        tool_name,
        description,
        context_type,
        fields,
        call_args,
        ..
    } = analyze(opts, &mut func.sig, &func.attrs, false)?;

    let args_ident = format_ident!("__{}Args", pascal_case(&fn_ident.to_string()));
    let ctor_ident = format_ident!("{}_tool", fn_ident);
    let is_async = func.sig.asyncness.is_some();
    // A synchronous function runs on the blocking pool, as Python runs it in a worker thread,
    // so it does not stall the runtime while it blocks.
    let call = if is_async {
        quote!(#fn_ident(#(#call_args),*).await)
    } else {
        quote!(#sdk::__private::run_blocking(move || #fn_ident(#(#call_args),*)).await)
    };
    let result = if returns_result(&func.sig.output) {
        quote!(#call.map_err(::core::convert::Into::<#sdk::AgentsError>::into))
    } else {
        quote!(::core::result::Result::<_, #sdk::AgentsError>::Ok(#call))
    };
    let ctx_pat = if context_type.is_some() {
        quote!(__ctx)
    } else {
        quote!(_)
    };
    let (generics, ctx_ty, where_clause) = match &context_type {
        Some(c) => (quote!(), quote!(#c), quote!()),
        None => (
            quote!(<__C>),
            quote!(__C),
            quote!(where __C: ::core::marker::Send + ::core::marker::Sync + 'static),
        ),
    };
    let strict = strict_opt
        .map(|s| quote!(.with_strict_json_schema(#s)))
        .unwrap_or_default();
    let ctor_doc = format!("Build the `{tool_name}` function tool from [`{fn_ident}`].");
    let schemars_path = format!("{sdk}::__private::schemars").replace(' ', "");
    let serde_path = format!("{sdk}::__private::serde").replace(' ', "");

    Ok(quote! {
        #func

        #[doc(hidden)]
        #[allow(non_camel_case_types, dead_code)]
        #[derive(#sdk::__private::serde::Deserialize, #sdk::__private::schemars::JsonSchema)]
        #[serde(crate = #serde_path)]
        #[schemars(crate = #schemars_path)]
        #vis struct #args_ident {
            #(#fields,)*
        }

        #[doc = #ctor_doc]
        #vis fn #ctor_ident #generics () -> #sdk::FunctionTool<#ctx_ty> #where_clause {
            #sdk::FunctionTool::<#ctx_ty>::new(
                #tool_name,
                #description,
                |#ctx_pat: #sdk::tool_context::ToolContext<#ctx_ty>, __args: #args_ident| async move {
                    #result
                },
            )
            #strict
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_doc_sections() {
        let lines: Vec<String> = [
            "Get the weather.",
            "",
            "# Arguments",
            "* `city` - The city.",
            "* `days` - Number of days",
            "  to forecast.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (d, a) = parse_docs(&lines);
        assert_eq!(d, "Get the weather.");
        assert_eq!(a[0], ("city".into(), "The city.".into()));
        // griffe keeps the line break of a wrapped argument description; so does the macro.
        assert_eq!(a[1].1, "Number of days\nto forecast.");
        let google: Vec<String> = ["Sum.", "Args:", "    a: first", "    b: second"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (d, a) = parse_docs(&google);
        assert_eq!(d, "Sum.");
        assert_eq!(
            a,
            vec![("a".into(), "first".into()), ("b".into(), "second".into())]
        );
        assert_eq!(pascal_case("get_weather"), "GetWeather");
    }
}
