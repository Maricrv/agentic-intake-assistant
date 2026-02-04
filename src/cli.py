import argparse
import json
import os

from agent.agent import GenericIntakeAgent

SESSIONS_DIR = "sessions"


def ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session(session_id: str) -> dict | None:
    path = session_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_session(session_id: str, data: dict) -> None:
    path = session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_intent_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Agentic intake assistant (config-driven).")
    parser.add_argument(
        "--config",
        default="configs/intents_generic.json",
        help="Path to the intent pack JSON config (default: configs/intents_generic.json)",
    )
    args = parser.parse_args()

    ensure_sessions_dir()

    cfg = load_intent_config(args.config)

    session_id = input("Session id (press Enter for sess_local_001): ").strip() or "sess_local_001"
    previous = load_session(session_id)

    agent = GenericIntakeAgent(
        request_id="req_local_000001",
        session_id=session_id,
        previous_state=previous,
        intent_config=cfg,
    )
    result = agent.run()

    print("\n--- intake_result (JSON) ---")
    out = result.to_dict()
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Save agent memory snapshot for next turn
    save_session(session_id, agent.export_state())


if __name__ == "__main__":
    main()
