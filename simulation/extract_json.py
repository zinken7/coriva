import json
import argparse


def main():
    parser = argparse.ArgumentParser(description='Extract branch or pre-intervention JSON data')
    parser.add_argument('--input', required=True, help='Main coriva_bench_examples.json')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--scenario', required=True,
                        help='Scenario name (e.g. high_speed_following)')
    parser.add_argument('--branch', help='Branch name (for intervention data)')
    parser.add_argument('--before-time', type=float,
                        help='Extract steps before this time in seconds (for test)')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    result = []
    for sc in data:
        if sc['scenario'] == args.scenario:
            if args.branch:
                # Get data for a specific branch
                result = sc['branches'].get(args.branch, [])
            elif args.before_time is not None:
                # Get data before intervention (use first branch since all are identical)
                first_branch_data = list(sc['branches'].values())[0]
                result = []
                for step in first_branch_data:
                    if step['time_sec'] < args.before_time:
                        # Remove branch name since pre-intervention data is shared
                        entry = {k: v for k, v in step.items() if k != 'branch'}
                        result.append(entry)
            break

    if not result:
        print(f"  [WARN] No data found for scenario={args.scenario}, "
              f"branch={args.branch}, before_time={args.before_time}")

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Extracted {len(result)} entries → {args.output}")


if __name__ == '__main__':
    main()
