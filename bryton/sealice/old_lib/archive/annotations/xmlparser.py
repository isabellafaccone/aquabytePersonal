
# *** MAKE SURE TO CHECK WHETHER POINTS ARE POLYGON OR LINE ***
import math
import re
import sys

# to represent a single annotation for a lice
class LiceAnnotation():
    def __init__(self, frame, numPoints, xCoords, yCoords, lice_flag):
        self.frame = frame
        self.numPoints = numPoints
        self.xCoords = xCoords
        self.yCoords = yCoords
        self.lice_flag = lice_flag
        
class dataFile():
    def __init__(self, filename):
        self.filename = filename
    # returns list of lists of LiceAnnotations
    def parseFile(self):
        with open(self.filename, "r") as f:
            # each element cooresponds to a unique lice
            # each element is a list of LiceAnnotations for that unique lice
            lice = []
            line = f.readline()
            while (line != ''):
                line = line.strip()
                # if in a new lice object
                lice_flag = -1
                if (line[0:7] == '<object'):
                    # each element corresponds to an annotation for the current lice
                    temp1 = line.split()
                    temp1 = temp1[3]
                    temp1 = temp1[6:-2]
                    print temp1
                    if 'Non-' in temp1 or 'Not-' in temp1:
                        lice_flag = 0
                    else:
                        lice_flag = 1

                    annotations = []

                    objectline = f.readline()
                    objectLine = objectline.strip()
                    while (objectline != '</object>'):
                        # if in an annotation
                        if (objectline[0:15] == '<data:polygone '):
                            words = objectline.split(' ')
                            #print words
                            framespan = words[-1]
                            framespan = re.sub('[framespan=>"]', '', framespan)
                            framespan = framespan.split(':')
                            frame = framespan[0]

                            # number of data points
                            numPoints = 0
                            # vertice coordinates
                            xCoords = []
                            yCoords = []

                            objectline = f.readline()
                            objectline = objectline.strip()
                            # if in a data point line
                            while (objectline[0:11] == '<data:point'):
                                numPoints += 1

                                data = objectline.split(' ')
                                x = int(re.sub('[x="]', '', data[1]))
                                xCoords.append(x)
                                y = int(re.sub('[y="/>]', '', data[2]))
                                yCoords.append(y)

                                objectline = f.readline()
                                objectline = objectline.strip()

                            annotations.append(LiceAnnotation(frame, numPoints, xCoords, yCoords, lice_flag))

                        objectline = f.readline()
                        objectline = objectline.strip()

                    lice.append(annotations)

                line = f.readline()

            return lice

    def toList(self,lice):
        liceList = []
        for l in lice:
            for annotation in l:
                tempList = []
                for i in range (0, len(annotation.xCoords)):
                    tempList.append((annotation.xCoords[i], annotation.yCoords[i]))
                liceList.append([int(annotation.frame), annotation.numPoints, tempList, annotation.lice_flag])
        return liceList
            
def main():
    # filename = raw_input('Enter the filename: ')
    filename = sys.argv[1]

    newFile = dataFile(filename)

    lice = newFile.parseFile()
    print lice 
    # for each unique lice in lice[]
    for l in lice:
        print l
        # for each annotation of the specific lice
        for annotation in l:
            # print frame number and number of data points in the annotation
            #print('Frame: %s, NumberOfPoints: %i\n' %(annotation.frame, annotation.numPoints))
            kjk=0
            # prints x and y values of each data point in the annotation
            for i in range (0, len(annotation.xCoords)):
                kjk=0
                #print('(%i, %i)\n' %(annotation.xCoords[i], annotation.yCoords[i]))

if __name__ == '__main__':
    main()

