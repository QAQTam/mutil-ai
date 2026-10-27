use mutil_ai::{CustomToolAccumulator, NativeToolIdentity, NativeToolPayload};

fn identity() -> NativeToolIdentity {
    NativeToolIdentity {
        namespace: Some("functions".into()),
        name: Some("apply_patch".into()),
        call_id: Some("call_1".into()),
        item_id: Some("ctc_1".into()),
    }
}

#[test]
fn fragments_and_snapshots_preserve_bytes_without_doubling() {
    let mut call = CustomToolAccumulator::new();
    call.merge_identity(&identity()).unwrap();
    let patch = "*** Begin Patch\n+你好 \"x\" \\ path\n+\n*** End Patch\n";
    call.push_delta("*** Begin Patch\n+你好 ").unwrap();
    call.push_delta("\"x\" \\ path\n+\n").unwrap();
    call.finish_input(patch).unwrap();
    assert!(call.completed_call().is_err());
    call.finish_item(patch).unwrap();
    call.finish_item(patch).unwrap();
    let completed = call.completed_call().unwrap();
    assert_eq!(completed.call_id, "call_1");
    assert_eq!(completed.item_id.as_deref(), Some("ctc_1"));
    assert_eq!(completed.payload, NativeToolPayload::RawText(patch.into()));
}

#[test]
fn conflicting_snapshots_and_identity_fail_without_mutation() {
    let mut call = CustomToolAccumulator::new();
    call.merge_identity(&identity()).unwrap();
    call.push_delta("hello").unwrap();
    assert!(call.finish_item("hellX").is_err());
    assert_eq!(call.input(), "hello");
    assert!(call.completed_call().is_err());

    let mut call = CustomToolAccumulator::new();
    call.merge_identity(&identity()).unwrap();
    let mut conflict = identity();
    conflict.call_id = Some("call_2".into());
    assert!(call.merge_identity(&conflict).is_err());
    assert!(call.completed_call().is_err());

    let mut call = CustomToolAccumulator::new();
    call.merge_identity(&identity()).unwrap();
    call.push_delta("hello").unwrap();
    call.finish_item("hello world").unwrap();
    assert_eq!(call.completed_call().unwrap().call_id, "call_1");
    assert!(call.finish_item("hello there").is_err());
    assert!(call.completed_call().is_err());
    assert!(call.push_delta("!").is_err());
}

#[test]
fn interruption_and_missing_identity_are_not_executable() {
    let mut call = CustomToolAccumulator::new();
    call.finish_item("*** End Patch").unwrap();
    assert!(call.completed_call().is_err());
    call.merge_identity(&identity()).unwrap();
    assert!(call.completed_call().is_ok());
    call.interrupt();
    assert!(call.completed_call().is_ok());
    let mut interrupted = CustomToolAccumulator::new();
    interrupted.merge_identity(&identity()).unwrap();
    interrupted.push_delta("*** End Patch").unwrap();
    interrupted.interrupt();
    assert!(interrupted.completed_call().is_err());
    assert!(interrupted.finish_item("*** End Patch").is_err());
}
