"""Interactive CLI loop for the HCMA coding assistant."""

import time

from hcma.agents.consolidation_agent import ConsolidationAgent
from hcma.agents.task_agent import TaskAgent
from hcma.config import settings
from hcma.memory.consolidation_loop import ConsolidationLoop
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore


def main() -> None:
    buf = EpisodicBuffer(
        db_path=settings.SQLITE_DB_PATH,
        capacity=settings.EPISODIC_BUFFER_CAPACITY,
    )
    ltm = LTMStore(
        db_path=settings.SQLITE_DB_PATH,
        qdrant_storage_path=settings.QDRANT_STORAGE_PATH,
        collection_name=settings.LTM_COLLECTION_NAME,
    )
    session_id = f"{settings.SESSION_ID_PREFIX}_{int(time.time())}"
    agent = TaskAgent(buffer=buf, session_id=session_id, ltm=ltm)
    consolidation_agent = ConsolidationAgent(buffer=buf, ltm=ltm)
    loop = ConsolidationLoop(
        buffer=buf,
        ltm=ltm,
        check_interval_seconds=settings.CONSOLIDATION_CHECK_INTERVAL,
    )

    loop.start()

    print(f"HCMA Coding Assistant — session: {session_id}")
    print(
        f"Buffer capacity: {settings.EPISODIC_BUFFER_CAPACITY}. "
        "Type 'quit' to exit, 'status' to see buffer state."
    )

    try:
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input == "quit":
                break

            if user_input == "status":
                count = buf.get_count()
                print(f"Buffer: {count}/{buf.capacity} entries")
                continue

            if user_input == "history":
                print(f"Conversation turns: {len(agent.get_conversation_history())} messages")
                continue

            if user_input == "ltm":
                memories = ltm.get_all()
                print(f"LTM memories: {len(memories)} total")
                for mem in memories[:3]:
                    print(f"  [{mem.id[:8]}] {mem.memory_type:<12} {mem.content[:60]}")
                continue

            if user_input == "contradictions":
                flags = ltm.get_unresolved_contradictions()
                if not flags:
                    print("No contradictions detected")
                else:
                    for flag in flags:
                        print(
                            f"[{flag.severity.upper()}] "
                            f"{flag.memory_id_a[:8]} vs {flag.memory_id_b[:8]}: "
                            f"{flag.reason}"
                        )
                continue

            if user_input == "consolidate":
                print("Running consolidation…")
                result = consolidation_agent.run()
                print(
                    f"  promoted={result.promoted} compressed={result.compressed} "
                    f"discarded={result.discarded} contradictions={result.contradictions_found} "
                    f"duration={result.duration_seconds:.2f}s"
                )
                continue

            if not user_input:
                continue

            response = agent.run(user_input)
            print(response)

            trigger_count = int(buf.capacity * settings.CONSOLIDATION_TRIGGER_RATIO)
            if buf.get_count() >= trigger_count:
                print(
                    f"WARNING: buffer at {buf.get_count()}/{buf.capacity} "
                    f"(≥{settings.CONSOLIDATION_TRIGGER_RATIO:.0%}) — "
                    "background consolidation will trigger shortly."
                )
    finally:
        loop.stop()

    print(f"Session ended. Final buffer count: {buf.get_count()} entries.")


if __name__ == "__main__":
    main()
