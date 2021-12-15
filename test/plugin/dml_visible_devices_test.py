import absl.testing.absltest as absltest
import os
from absl import flags

flags.DEFINE_string("dml_visible_devices", "", "Value of DML_VISIBLE_DEVICES environment variable")

class VisibleDevicesTest(absltest.TestCase):
    def test(self):
        os.environ["DML_VISIBLE_DEVICES"] = flags.FLAGS.dml_visible_devices

        # See https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq
        # The value should be a comma-separated list of device IDs. 
        # Any IDs appearing after -1 are invalid.
        valid_id_count = 0
        for id in flags.FLAGS.dml_visible_devices.split(','):
            if id == "-1":
                break
            valid_id_count += 1

        import tensorflow as tf
        dml_devices = tf.config.list_physical_devices("DML")

        # We can't guarantee the machine running this test has multiple devices/adapters,
        # but it must have at least one.
        if valid_id_count == 0:
            self.assertEmpty(dml_devices)
        else:
            self.assertBetween(len(dml_devices), 1, valid_id_count)

if __name__ == "__main__":
    absltest.main()