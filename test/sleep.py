import time
import absl.testing.absltest as absltest
from absl import flags

flags.DEFINE_string("seconds", "", "Value seconds")

class SleepTest(absltest.TestCase):
    def test(self):
        time.sleep(int(flags.FLAGS.seconds))

if __name__ == "__main__":
    absltest.main()