import absl.testing.absltest as absltest
import time

class SlowTest(absltest.TestCase):
    def testSleep(self):
        time.sleep(10)

if __name__ == "__main__":
    absltest.main()