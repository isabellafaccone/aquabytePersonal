# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:19:41 2018

@author: sramena1
"""

import sys
import xml.etree.ElementTree
from operator import itemgetter

class Annotation(object):
	def __init__(self, frame_number, points):
		self.frame_number = frame_number
		self.points = points

class Entity(object):
	def __init__(self, name):
		self.name = name
		self.annotations = []
		self.type = None
		self.determine_entity_type()

	def determine_entity_type(self):
		if ('Non' in self.name or 'Not' in self.name):
			self.type = 'not_lice'
		else:
			self.type = 'lice'

class Parser(object):
    def __init__(self, xml_filename):
        self.xml_filename = xml_filename
        
    def parse(self):
        root = xml.etree.ElementTree.parse(self.xml_filename).getroot()
        prefix = '{http://lamp.cfar.umd.edu/viper}'
        prefix_data = '{http://lamp.cfar.umd.edu/viperdata}'
        sealice = []
        for child in root:
		child_tag = child.tag[len(prefix):]
		if child_tag == 'config':
			entities = {}
			for entity in child:
				entity_name = entity.attrib['name']
				entities[entity_name] = Entity(entity_name)

		elif child_tag == 'data':
			for subchild in child:
				subchild_tag = subchild.tag[len(prefix):]
				if subchild_tag == 'sourcefile':
					for sub2child in subchild:
						sub2child_tag = sub2child.tag[len(prefix):]
						if sub2child_tag == 'object':
							for sub3child in sub2child:
								sub3child_tag = sub3child.tag[len(prefix):]
								if sub3child_tag == 'attribute':
									this_entity = entities[sub3child.attrib['name']]
									for sub4child in sub3child:
										sub4child_tag = sub4child.tag[len(prefix_data):]
										if sub4child_tag == 'polygone':
											frame_number = sub4child.attrib['framespan'].split(':')[0]
											points = []
											for sub5child in sub4child:
												sub5child_tag = sub5child.tag[len(prefix_data):]
												if sub5child_tag == 'point':
													point_attrib = sub5child.attrib
													point = (int(point_attrib['x']), int(point_attrib['y']))
													points.append(point)
											annotation = Annotation(frame_number, points)
											this_entity.annotations.append(annotation)

        
        for name, entity in entities.iteritems():
            #print 'Name: {}'.format(name)
            #print 'Type: {}'.format(entity.type)

            for annotation in entity.annotations:
                #print '	Name: {}, Type: {}, Frame number: {}, Annotation: {}, Number of points: {}'.format(name, entity.type, annotation.frame_number, annotation.points, len(annotation.points))
                sealice.append([int(annotation.frame_number), len(annotation.points), annotation.points, entity.type])
        #print len(sealice)
        sorted_licelist = sorted(sealice,key = itemgetter(0))
        print len(sorted_licelist)
        return sorted_licelist

        
def main():
    # file_name = sys.argv[1]
    file_name = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_22_Lice_VIPER.xml"
    parser = Parser(file_name)
    sorted_licelist = parser.parse() 
    #print len(sorted_licelist)
    
if __name__ == '__main__':
	main()
