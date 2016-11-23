import glob

if __name__ == "__main__":
	filename = 'data/abortion/*.data'
	data = [open(f) for f in glob.glob(filename)]
	print len(data)
