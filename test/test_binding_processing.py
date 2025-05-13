import os 
import re
import unittest
from htbam_analysis.processing import chipcollections as collections


class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_process(self):
        root = './test/test_data/'
        binding_path = '{}/binding_dummy'.format(root)
        bait_channel, bait_exposure = '2', 5
        prey_channel, prey_exposure = '3', 5

        # make a dummy processor object to test file grabbing
        binding_processor = collections.ButtonBindingSeries(
            device=None, 
            button_ref=None, 
            prey_channel=prey_channel,
            prey_exposure=prey_exposure,
            bait_channel=bait_channel,
            bait_exposure=bait_exposure
            )

        def concentration_from_file(file):

            parent_path = binding_path

            # use regex to pull out note
            handle = os.path.relpath(file, parent_path).split('/')[0]
            note_match = re.search(r"\d{8}-\d{6}-d\d+_(.+?)_(PreWash|PostWash)_Quant", handle)
            if note_match:
                note = note_match.group(1)
            else:
                pass

            concentration = float(note.split('_ADPR_100AF')[0].replace('_', '.'))
            return concentration

        # grab files and concentrations
        binding_processor.grab_binding_images(binding_path, verbose=False, concentration_parser=concentration_from_file)

        # unit testing for file grabbing and concentration parsing
        image_batches = [binding_processor.prewash_bait_images, binding_processor.postwash_bait_images, binding_processor.postwash_prey_images]
        concentration_batches = [binding_processor.prewash_bait_concentrations, binding_processor.postwash_bait_concentrations, binding_processor.postwash_prey_concentrations]

        # check that total images is equivalend to total concentrations
        total_images = sum([len(i) for i in image_batches])
        self.assertEqual(sum(len(c) for c in concentration_batches), total_images)

        for images, concentrations in zip(
            image_batches,
            concentration_batches
        ):
            self.assertEqual(len(images), len(concentrations))
            self.assertEqual(len(images) % 3, 0)
            self.assertEqual(total_images / len(images), 3)

            for image, concentration in zip(images, concentrations):
                parsed_concentration = float(image.split('/')[5].split('d3_')[1].split('_ADPR')[0].replace('_', '.'))
                self.assertEqual(parsed_concentration, concentration)
        

if __name__ == '__main__':
    unittest.main()