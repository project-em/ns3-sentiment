import glob

if __name__ == "__main__":
	filename = 'data/abortion/*.data'
	data = glob.glob(filename)
	print len(data)
