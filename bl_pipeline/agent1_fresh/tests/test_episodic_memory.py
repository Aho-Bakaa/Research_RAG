"""Unit tests for Phase 6: EpisodicMemory and LangfuseTracer.
"""
from bl_pipeline.rag.episodic_memory import DiagnosticMemory, EpisodicMemoryManager
from bl_pipeline.shared.langfuse_tracer import trace_span


def test_episodic_memory_record_and_retrieve():
    manager = EpisodicMemoryManager()

    diag = DiagnosticMemory(
        case_id="LT1_wind_tunnel_IISc",
        symptom_type="length_scale_mismatch",
        velocity_ms=6.2,
        tu_pct=2.7,
        root_cause="Assumed Roach far-field Lambda_x=12.8mm, but true measured Lambda_x was 39.4mm",
        lesson_learned="When Lambda_x is elevated outside FS20 calibration band, transition onset moves downstream.",
        suggested_fix="Check hot-wire autocorrelation zero-crossing Lambda_x before applying FS20.",
        citation="dubey_2026_thesis",
    )

    mem_id = manager.record_diagnostic(diag)
    assert mem_id == diag.memory_id

    # Retrieve with filter
    retrieved = manager.retrieve_relevant_memories(symptom_type="length_scale_mismatch")
    assert len(retrieved) == 1
    assert "39.4mm" in retrieved[0]
    assert "[HISTORICAL CASE LESSON" in retrieved[0]

    # Non-matching symptom filter returns empty
    non_match = manager.retrieve_relevant_memories(symptom_type="unrelated_symptom")
    assert len(non_match) == 0

    print("PASS  test_episodic_memory_record_and_retrieve")


def test_langfuse_offline_noop():
    with trace_span("test_span", metadata={"stage": "retrieval"}) as span:
        # Should gracefully succeed without raising
        pass
    print("PASS  test_langfuse_offline_noop")


if __name__ == "__main__":
    test_episodic_memory_record_and_retrieve()
    test_langfuse_offline_noop()
    print("------------------------------------------------------------")
    print("Phase 6 episodic memory and observability tests passed!")
