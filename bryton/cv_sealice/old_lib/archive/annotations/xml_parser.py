import sys
import xml.etree.ElementTree

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


def main():
	file_name = sys.argv[1]
	root = xml.etree.ElementTree.parse(file_name).getroot()

	prefix = '{http://lamp.cfar.umd.edu/viper}'
	prefix_data = '{http://lamp.cfar.umd.edu/viperdata}'
	
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
													point = (point_attrib['x'], point_attrib['y'])
													points.append(point)
											annotation = Annotation(frame_number, points)
											this_entity.annotations.append(annotation)


	for name, entity in entities.iteritems():
		print 'Name: {}'.format(name)
		print 'Type: {}'.format(entity.type)
		for annotation in entity.annotations:
			print '	Frame number: {}, Annotation: {}, Number of points: {}'.format(annotation.frame_number, annotation.points, len(annotation.points))



if __name__ == '__main__':
	main()
