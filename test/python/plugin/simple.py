import absl.testing.absltest as absltest

class MyTest(absltest.TestCase):
    def testFoo(self):
        self.assertTrue(False)

    def testAlpha(self):
        self.assertTrue(True)

    def testBravo(self):
        self.assertTrue(True)

if __name__ == "__main__":
    absltest.main()