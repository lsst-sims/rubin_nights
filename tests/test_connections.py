import os
import unittest
from pathlib import Path

import rubin_nights.connections as connections


class TestConnections(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(__file__).parent
        self.tokenfile = os.path.join(self.test_dir, "data", "dummy_token")
        # This matches content from dummy_token file
        self.expected_token = "NotARealToken"
        # This is just for an environment token
        self.expected_token2 = "NotARealTokenEither"

    def test_access_token(self) -> None:
        # Save current environment settings
        current_env_token = os.getenv("ACCESS_TOKEN")
        current_env_tokenfile = os.getenv("ACCESS_TOKEN_FILE")
        # Replace them with nothing
        if current_env_token is not None:
            del os.environ["ACCESS_TOKEN"]
        if current_env_tokenfile is not None:
            del os.environ["ACCESS_TOKEN_FILE"]

        # Use the test tokenfile
        token = connections.get_access_token(tokenfile=self.tokenfile)
        self.assertEqual(token, self.expected_token)
        # Use nothing (and use bad default_tokenfile
        token = connections.get_access_token(default_tokenfile="NoToken")
        if os.getenv("EXTERNAL_INSTANCE_URL") is None:
            self.assertTrue(len(token) == 0)
        # This is expected to get a real token on the RSP
        else:
            self.assertTrue(len(token) > 0)
        # Use environment variable as token
        os.environ["ACCESS_TOKEN"] = self.expected_token2
        token = connections.get_access_token()
        self.assertEqual(token, self.expected_token2)
        del os.environ["ACCESS_TOKEN"]
        # Use environment variable for tokenfile
        os.environ["ACCESS_TOKEN_FILE"] = self.tokenfile
        token = connections.get_access_token()
        self.assertEqual(token, self.expected_token)
        del os.environ["ACCESS_TOKEN_FILE"]
        # Reset pre-environment variables.
        if current_env_token is not None:
            os.environ["ACCESS_TOKEN"] = current_env_token
        if current_env_tokenfile is not None:
            os.environ["ACCESS_TOKEN_FILE"] = current_env_tokenfile

    def test_endpoints(self) -> None:
        # Check definition of some sites
        for site in ["usdf", "usdf-dev", "summit", "base"]:
            tokenfile = os.path.join(self.test_dir, self.tokenfile)
            endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
            self.assertTrue(isinstance(endpoints["api_base"], str))
        # Check expected clients are added to the dictionary
        clients = ["consdb", "consdb_tap", "efd", "obsenv", "narrative_log", "exposure_log", "night_report"]
        endpoint_keys = list(endpoints.keys())
        self.assertTrue(len([c for c in clients if c not in endpoint_keys]) == 0)

    def test_usdf_lfa(self) -> None:
        http_uri = "https://s3.cp.lsst.org/"
        uri = (
            "rubinobs-lfa-cp/Scheduler:1/"
            "Scheduler:1/2025/07/21/Scheduler:1_Scheduler:1_2025-07-22T03:05:04.297.p"
        )
        result = (
            "s3://lfa@rubinobs-lfa-cp/Scheduler:1/"
            "Scheduler:1/2025/07/21/Scheduler:1_Scheduler:1_2025-07-22T03:05:04.297.p"
        )
        new_uri = connections.usdf_lfa(http_uri + uri, bucket="s3://lfa@")
        self.assertEqual(new_uri, result)


if __name__ == "__main__":
    unittest.main()
