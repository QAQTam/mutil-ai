use mutil_ai::{
    AuthStyle, EndpointSpec, MaxTokensSemantics, ModelMatcher, ModelProfile, ProfileId,
    ProfileRegistry, ProfileSelector, ProtocolSurface, ProviderProfile, ReasoningReplayPolicy,
    ThinkingRequestProfile,
};

#[test]
fn endpoint_requires_explicit_protocol_and_substitutes_model() {
    let endpoint = EndpointSpec::gemini_generate_content("https://example.test/v1beta");

    assert_eq!(endpoint.protocol, ProtocolSurface::GeminiGenerateContent);
    assert_eq!(
        endpoint.url("gemini-3-flash").unwrap(),
        "https://example.test/v1beta/models/gemini-3-flash:generateContent"
    );
    assert!(matches!(
        endpoint.auth,
        AuthStyle::QueryKey { ref parameter } if parameter == "key"
    ));
}

#[test]
fn endpoint_never_infers_a_protocol_from_the_url() {
    let endpoint = EndpointSpec::new(
        ProtocolSurface::OpenAiChat,
        "https://gateway.example.test/anthropic",
        "/v1/messages",
        AuthStyle::None,
    );

    assert_eq!(endpoint.protocol, ProtocolSurface::OpenAiChat);
    assert_eq!(
        endpoint.url("model-a").unwrap(),
        "https://gateway.example.test/anthropic/v1/messages"
    );
}

#[test]
fn profile_selector_is_explicit() {
    let profile = ProviderProfile::new(ProfileId::from("company-gateway"));
    let endpoint = EndpointSpec::openai_chat("https://example.test/v1").profile(profile);

    assert!(matches!(endpoint.profile, ProfileSelector::Custom(_)));
    assert_eq!(
        endpoint.provider_profile().unwrap().id.as_str(),
        "company-gateway"
    );
}

#[test]
fn model_profiles_use_exact_and_prefix_matching() {
    let exact = ModelProfile {
        matcher: ModelMatcher::Exact("kimi-k2".to_string()),
        thinking: ThinkingRequestProfile::EnabledFlag("thinking".to_string()),
        replay: Some(ReasoningReplayPolicy::SameProvider),
        max_tokens_semantics: MaxTokensSemantics::MaxTokens,
        ..ModelProfile::default()
    };

    assert!(exact.matches("kimi-k2"));
    assert!(!exact.matches("kimi-k2.5"));

    let prefix = ModelProfile {
        matcher: ModelMatcher::Prefix("qwen3".to_string()),
        ..ModelProfile::default()
    };
    assert!(prefix.matches("qwen3-235b"));
}

#[test]
fn profile_registry_resolves_explicit_ids_and_aliases() {
    let registry = ProfileRegistry::new();

    let kimi = registry
        .resolve(&ProfileId::from("kimi"))
        .expect("kimi profile");
    assert_eq!(kimi.id.as_str(), "kimi");

    let moonshot = registry
        .resolve(&ProfileId::from("moonshot"))
        .expect("moonshot alias");
    assert_eq!(moonshot.id.as_str(), "kimi");

    assert!(registry.resolve(&ProfileId::from("unknown")).is_none());
    assert!(registry.builtin_ids().contains(&"deepseek"));
}

#[test]
fn endpoint_resolves_builtin_profile_without_changing_protocol() {
    let endpoint = EndpointSpec::new(
        ProtocolSurface::OpenAiChat,
        "https://gateway.example/v1",
        "/chat/completions",
        AuthStyle::Bearer,
    )
    .profile_selector(ProfileSelector::Builtin(ProfileId::from("qwen")));

    let profile = endpoint.resolved_profile().expect("builtin profile");
    assert_eq!(profile.id.as_str(), "qwen");
    assert_eq!(endpoint.protocol, ProtocolSurface::OpenAiChat);
    assert_eq!(endpoint.path, "/chat/completions");
}
