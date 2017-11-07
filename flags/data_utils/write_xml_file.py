import os

def write_xml_file(boxes, labels, file_size, file_path, save_folder, truncated_array, difficult_array, occluded_array):
	file_name = file_path.split('/')[-1]
	base_file_name = file_name.split('.')[0]
	xml_file_name = '{}.xml'.format(base_file_name)
	with open(os.path.join(save_folder, xml_file_name), 'w') as xml_file:
		xml_file.write('<annotation>\n')
		xml_file.write('\t<folder>{}</folder>\n'.format(save_folder))
		xml_file.write('\t<filename>{}.png</filename>\n'.format(base_file_name))
		xml_file.write('\t<path>{}</path>\n'.format(file_path))

		xml_file.write('\t<source>\n')
		xml_file.write('\t\t<database>Unknown</database>\n')
		xml_file.write('\t</source>\n')

		xml_file.write('\t<size>\n')
		xml_file.write('\t\t<width>{}</width>\n'.format(file_size[1]))
		xml_file.write('\t\t<height>{}</height>\n'.format(file_size[0]))
		xml_file.write('\t\t<depth>{}</depth>\n'.format(file_size[2]))
		xml_file.write('\t</size>\n')
		xml_file.write('\t<segmented>0</segmented>\n')

		for box, label, truncated, difficult, occlusion in zip(boxes, labels, truncated_array,
																difficult_array, occluded_array):
			xml_file.write('\t<object>\n')
			xml_file.write('\t\t<name>{}</name>\n'.format(label))
			xml_file.write('\t\t<pose>Frontal</pose>\n')
			xml_file.write('\t\t<truncated>{}</truncated>\n'.format('1' if truncated else '0'))
			xml_file.write('\t\t<difficult>{}</difficult>\n'.format('1' if difficult else '0'))
			xml_file.write('\t\t<occluded>{}</occluded>\n'.format('1' if occlusion else '0'))

			xml_file.write('\t\t<bndbox>\n')
			xml_file.write('\t\t\t<xmin>{}</xmin>\n'.format(box[0]))
			xml_file.write('\t\t\t<xmax>{}</xmax>\n'.format(box[2]))
			xml_file.write('\t\t\t<ymin>{}</ymin>\n'.format(box[1]))
			xml_file.write('\t\t\t<ymax>{}</ymax>\n'.format(box[3]))
			xml_file.write('\t\t</bndbox>\n')

			xml_file.write('\t</object>\n')
            
		xml_file.write('</annotation>')