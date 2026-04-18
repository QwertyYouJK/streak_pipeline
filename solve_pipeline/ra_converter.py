import sys


def main():
    if len(sys.argv) < 2:
        print("Please input RA in hours, minutes, and seconds OR just in degrees")

    if len(sys.argv) == 4:
        h = sys.argv[1]
        m = sys.argv[2]
        s = sys.argv[3]

        deg = 15 * (float(h) + float(m) / 60 + float(s) / 3600)
        print(f"RA in {h} {m} {s} = {deg} degrees")
    elif len(sys.argv) == 2:
        deg = sys.argv[1]

        h = float(deg) / 15.0
        m = (h % 1) * 60
        s = (m % 1) * 60
        print(f"RA in {deg} degrees = {h // 1} {m // 1} {s}")


if __name__ == "__main__":
    main()
