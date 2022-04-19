import os
import argparse
import zipfile


def ftw(path):
    for cur, _dirs, files in os.walk(path):
        for file in files:
            yield os.path.join(cur, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Export all s2meta into a zip file")
    parser.add_argument("input_file", nargs=1, help="Directory root")
    parser.add_argument('--json', dest="json", help="Rename .s2meta to .json", action="store_true")
    args = parser.parse_args()

    INPUT_FILE = args.input_file[0]
    RENAME_JSON = args.json

    archive = zipfile.ZipFile("metadata.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)

    for filename in ftw(INPUT_FILE):
        if filename.endswith(".s2meta"):

            relpath = os.path.relpath(filename, INPUT_FILE)

            if RENAME_JSON:
                relpath = relpath.replace(".s2meta", ".json")

            archive.write(filename, relpath)

    archive.close()


