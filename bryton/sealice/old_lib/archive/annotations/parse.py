def readQuote(self, filename):
	# disregarding first quote
	filename.read(1)
	chars = []
	c = filename.read(1)
	#reads up until closing " for x, y, width, height and up until : for framespan
	while (c != '"' and c != ':'):
		chars.append(c)
		c = filename.read(1)
	word = "".join(chars)
	return word

def main():
	filename = raw_input('Enter the filename: ')
	outputFile = open(filename+'_parsed', 'w')
	with open(filename, "r") as f:
		for line in f:
			# removing all spaces/tabs
			line = line.strip()
			# if in a data section
			if (f.read(14) == '<data:polygone'):
				# discarding rest of header line
				f.readline()
				# discarding 'framespan =' and storing fame value
				f.read(10)
				frame = int(readQuote(f))
				# moving to coordinates line
				f.readline()

				# vertice coordinates
				xCoords = []
				yCoords = []
				c = f.read(11)
				while (c == '<data:point'):
					# removing ' x=' and storing value of x
					f.read(3)
					x = int(readQuote(f))
					xCoords.append(x)
					# removing ' y=' and storing value of y
					f.read(3)
					y = int(readQuote(f))
					yCoords.append(y)
					# moving to next line
					f.readline()
					c = f.read(11)
				f.readline()

				# removing ' width=' and storing value of width
				f.read(7)
				width = int(readQuote(f))
				# removing ' height=' and storing value of height
				f.read(8)
				height = int(readQuote(f))
				# removing ' framespan=' and storing frame value
				f.read(11)
				frame = int(readQuote(f))

				# calculating midpoint in x direction
				#xMid = 
				# calculating midpoint in y direction
				#yMid = 

				# edge midpoint coordinates
				midpointsX = []
				midpointsY = []
				# calculating length
				# first calculating all edge midpoints
				# then calculating every possible distance between each of the midpoints
				# using the largest distance as length
				length = 0;
				for i in range(0, len(xCoords)):
					x = (xCoords[i]+xCoords[(i+1)%len(xCoords)])/2.0
					midpointsX.append(x)
					y = (yCoords[i]+yCoords[(i+1)%len(yCoords)])/2.0
					midpointsX.append(y)
				for i in range(0, len(midpointsX)):
					for j in range(i+1, len(midpointsX)):
						dist = sqrt((midpointsX[i] - midpointsX[j])**2 + (midpointsY[i] - midpointsY[j])**2)
						if (dist > length):
							length = dist;

				# calculating orientation
				#orientation = 

				# calculating polygon area
				# using crucial assumption that points were made in order around the polygon
				n =0
				for i in range(0, len(xCoords) - 1):
					n += (xCoords[i]*yCoords[i+1] - xCoords[i+1]*yCoords[i])
				area = abs(n/2.0)

				outputFile.write('frame: %i, xMid: %i, yMid: %i, length: %i, orientation: %i, area: %i\n' %(frame, xMid, yMid, length, orientation, area))
		f.close()

