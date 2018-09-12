import setuptools

setuptools.setup(
    name='monodepth',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six', 'scipy'],
    entry_points = {
        'console_scripts': [
            'retinanet-train=keras_retinanet.bin.train:main',
            'retinanet-evaluate-coco=keras_retinanet.bin.evaluate_coco:main',
            'retinanet-evaluate=keras_retinanet.bin.evaluate:main',
            'retinanet-debug=keras_retinanet.bin.debug:main',
            'retinanet-convert-model=keras_retinanet.bin.convert_model:main',
        ],
    }
)
