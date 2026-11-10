from .agent import Agent
from .config.settings import get_settings
from .llm import make_client
from .tools.tools import build_tools


def main():
    cfg = get_settings()
    llm = make_client(cfg)

    try:
        llm.ensure_ready()
    except RuntimeError as e:
        print(e)
        return

    agent = Agent(llm=llm, tools=build_tools(cfg), config=cfg)

    print(f"code-mentor [{cfg.LLM_PROVIDER}] ({cfg.MODEL}), проект: {cfg.ROOT}. Выход: exit")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        try:
            print(agent.ask(q))
        except ConnectionError:
            print(f"Потеряно соединение с {cfg.LLM_PROVIDER}. Проверьте сервер или сеть и повторите вопрос.")
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()