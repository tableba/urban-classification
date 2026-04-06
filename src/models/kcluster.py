from dataset.read_data import ReadTifs

reader = ReadTifs()


for name, t, data in reader.loop_through_files():
    return
    # do something
