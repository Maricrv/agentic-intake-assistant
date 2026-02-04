import argparse
import json
import os

from agent.agent import GenericIntakeAgent

SESSIONS_DIR = "sessions"


def ensure_sessions_dir() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session(session_id: str) -> dict | None:
    path = session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: session file is not valid JSON: {path}. Starting a fresh session.")
        return None


def save_session(session_id: str, data: dict) -> None:
    ensure_sessions_dir()
    path = session_path(session_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_intent_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file is not valid JSON: {config_path}. Error: {e}") from e

    if not cfg:
        raise ValueError(f"Config file is empty: {config_path}")

    if "intents" not in cfg or not isinstance(cfg.get("intents"), list):
        raise ValueError(f"Config missing 'intents' list: {config_path}")

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic intake assistant (config-driven).")
    parser.add_argument(
        "--config",
        default="configs/intents_generic.json",
        help="Path to the intent pack JSON config (default: configs/intents_generic.json)",
    )
    args = parser.parse_args()

    ensure_sessions_dir()

    try:
        cfg = load_intent_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    session_id = input("Session id (press Enter for sess_local_001): ").strip() or "sess_local_001"
    previous = load_session(session_id)

    agent = GenericIntakeAgent(
        request_id="req_local_000001",
        session_id=session_id,
        previous_state=previous,
        intent_config=cfg,
    )

    try:
        result = agent.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Saving session state...")
        save_session(session_id, agent.export_state())
        return 130
    except Exception as e:
        print(f"\nError during run: {e}\nSaving session state...")
        save_session(session_id, agent.export_state())
        return 1

    print("\n--- intake_result (JSON) ---")
    out = result.to_dict()
    print(json.dumps(out, indent=2, ensure_ascii=False))

    save_session(session_id, agent.export_state())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
