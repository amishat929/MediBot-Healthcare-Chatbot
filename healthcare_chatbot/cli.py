"""
Terminal version of the AI Healthcare Chatbot.   Run:  python cli.py
Type 'quit' to leave.
"""
import re
from chatbot import HealthBot

BOLD, DIM, RED, TEAL, END = "\033[1m", "\033[2m", "\033[91m", "\033[96m", "\033[0m"


def render(text: str) -> str:
    text = re.sub(r"^### (.*)$", rf"{BOLD}{TEAL}\1{END}", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", rf"{BOLD}\1{END}", text)
    text = re.sub(r"(?<![\w*])[*_](.+?)[*_](?![\w*])", rf"{DIM}\1{END}", text)
    return text


def main():
    bot = HealthBot()
    sid = "cli"
    print(f"{BOLD}{TEAL}MediBot - AI Healthcare Chatbot{END}  (type 'quit' to exit)")
    print(f"{DIM}General information only - not a diagnosis. In an emergency call 112 / 108.{END}\n")
    while True:
        try:
            msg = input(f"{BOLD}You > {END}").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if msg.lower() in ("quit", "exit", "q"):
            print("MediBot > Take care! 🌿"); break
        if not msg:
            continue
        r = bot.respond(sid, msg)
        colour = RED if r.get("emergency") else ""
        print(f"\n{colour}MediBot >{END} {render(r['reply'])}")
        if r.get("quick_replies"):
            print(f"{DIM}   [ " + " | ".join(r["quick_replies"]) + f" ]{END}")
        print()


if __name__ == "__main__":
    main()
