import absl.testing.absltest as absltest
import time

class ThisTestFails(absltest.TestCase):
    def testCaseA(self):
        self.assertFalse(True)

if __name__ == "__main__":
    absltest.main()