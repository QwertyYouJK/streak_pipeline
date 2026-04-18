import sys


def main():
    if len(sys.argv) < 2:
        print("Please input RA Dec in degrees")

    if len(sys.argv) == 5:
        if sys.argv[1] == "ra":
            h = sys.argv[2]
            m = sys.argv[3]
            s = sys.argv[4]

            deg = 15 * (float(h) + float(m) / 60 + float(s) / 3600)
            print(f"RA in {h} {m} {s} = {deg} degrees")
        elif sys.argv[1] == "dec":
            h = float(sys.argv[2])
            if h < 0:
                sign = "-"
            else:
                sign = ""
            m = float(sys.argv[3])
            s = float(sys.argv[4])

            deg = abs(h) + m / 60 + s / 3600
            print(f"RA in {h} {m} {s} = {sign}{deg} degrees")

    elif len(sys.argv) == 2:
        deg = sys.argv[1]

        h = float(deg) / 15.0
        m = (h % 1) * 60
        s = (m % 1) * 60
        print(f"RA in {deg} degrees = {h // 1} {m // 1} {s}")
    elif len(sys.argv) == 3:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])

        ra_hours_total = ra / 15.0
        rah = int(ra_hours_total)
        ram_total = (ra_hours_total - rah) * 60
        ram = int(ram_total)
        ras = (ram_total - ram) * 60

        print(f"RA in {ra} degrees = {rah}h {ram}m {ras:.3f}s")

        sign = "-" if dec < 0 else "+"
        dec_abs = abs(dec)

        decd = int(dec_abs)
        decm_total = (dec_abs - decd) * 60
        decm = int(decm_total)
        decs = (decm_total - decm) * 60

        print(f"Dec in {dec} degrees = {sign}{decd}d {decm}m {decs:.3f}s")


if __name__ == "__main__":
    main()
