from graph import run_research


def main():
    print("A-Share Multi-Agent Research System")
    print("Enter a Chinese company name or A-share stock code.")
    print("Examples: 贵州茅台, 茅台, 600519")
    print("Type 'exit' to quit.")
    print()

    while True:
        query = input("Stock: ").strip()

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Goodbye.")
            break

        if not query:
            continue

        print()
        print("Researching...")
        print()

        try:
            result = run_research(query)

            company_name = result.get(
                "company_name",
                query,
            )

            stock_code = result.get(
                "stock_code",
                "",
            )

            print(
                f"Company: {company_name}"
            )

            if stock_code:
                print(
                    f"Stock Code: {stock_code}"
                )

            print()
            print(
                result.get(
                    "final_report",
                    "No report generated.",
                )
            )

            errors = result.get(
                "errors",
                [],
            )

            if errors:
                print()
                print("Data warnings:")

                for error in errors:
                    print(
                        f"- {error}"
                    )

        except Exception as exc:
            print(
                f"Research failed: {exc}"
            )

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()