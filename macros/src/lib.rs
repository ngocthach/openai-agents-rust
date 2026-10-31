//! Procedural macros for `openai-agents-sdk`. Use them through the `macros` feature of the main
//! crate: `openai_agents_sdk::function_tool`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Expr, ExprLit, FnArg, GenericArgument, ItemFn, Lit, Meta, Pat, PathArguments,
    ReturnType, Token, Type, parse_macro_input,
};

/// Options of `#[function_tool(...)]`.
#[derive(Default)]
struct ToolOptions {
    name: Option<String>,
    description: Option<String>,
    strict: Option<bool>,
}

impl Parse for ToolOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut opts = ToolOptions::default();
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        for meta in metas {
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
/// rustdoc (`# Arguments` with `* \`name\` - text`) and Google style (`Args:` with
/// `name: text`) sections.
fn parse_docs(lines: &[String]) -> (String, Vec<(String, String)>) {
    let mut description = Vec::new();
    let mut args: Vec<(String, String)> = Vec::new();
    let mut in_args = false;
    for line in lines {
        let t = line.trim();
        let lower = t.to_ascii_lowercase();
        if lower == "# arguments" || lower == "# args" || lower == "args:" || lower == "arguments:"
        {
            in_args = true;
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
        description.push(line.clone());
    }
    let description = description.join("\n").trim().to_string();
    (description, args)
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
#[proc_macro_attribute]
pub fn function_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let opts = parse_macro_input!(attr as ToolOptions);
    let mut func = parse_macro_input!(item as ItemFn);
    match expand(opts, &mut func) {
        Ok(t) => t.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(opts: ToolOptions, func: &mut ItemFn) -> syn::Result<TokenStream2> {
    let sdk = quote!(::openai_agents_sdk);
    let fn_ident = func.sig.ident.clone();
    let vis = func.vis.clone();
    let (doc_description, doc_args) = parse_docs(&doc_lines(&func.attrs));
    let tool_name = opts.name.unwrap_or_else(|| fn_ident.to_string());
    let description = opts.description.unwrap_or(doc_description);

    let mut context_type: Option<Type> = None;
    let mut fields = Vec::new();
    let mut call_args = Vec::new();
    for (i, input) in func.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(pt) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "#[function_tool] does not support methods",
            ));
        };
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

    let args_ident = format_ident!("__{}Args", pascal_case(&fn_ident.to_string()));
    let ctor_ident = format_ident!("{}_tool", fn_ident);
    let is_async = func.sig.asyncness.is_some();
    let call = if is_async {
        quote!(#fn_ident(#(#call_args),*).await)
    } else {
        quote!(#fn_ident(#(#call_args),*))
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
    let strict = opts
        .strict
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
