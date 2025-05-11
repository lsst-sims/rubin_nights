import os
import unittest
from pathlib import Path

import rubin_nights.connections as connections


class TestConnections(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(__file__).parent
        self.tokenfile = "dummy_token"
        self.expected_token = "NotARealToken"

    def test_access_token(self) -> None:
        # Use the tokenfile
        token = connections.get_access_token(tokenfile=os.path.join(self.test_dir, self.tokenfile))
        self.assertEqual(token, self.expected_token)
        # Use nothing
        token = connections.get_access_token()
        if os.getenv("EXTERNAL_INSTANCE_URL") is None:
            self.assertTrue(token is None)
        # This is expected to get a real token on the RSP
        else:
            self.assertTrue(token is not None)
        # Use environment variable
        current_env_val = os.getenv("ACCESS_TOKEN")
        os.environ["ACCESS_TOKEN"] = self.expected_token
        token = connections.get_access_token()
        self.assertEqual(token, self.expected_token)
        if current_env_val is not None:
            os.environ["ACCESS_TOKEN"] = current_env_val

    def test_endpoints(self) -> None:
        # Check definition of some sites
        site = "usdf"
        tokenfile = os.path.join(self.test_dir, self.tokenfile)
        endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
        self.assertEqual(endpoints["api_base"], "https://usdf-rsp.slac.stanford.edu")
        site = "usdf-dev"
        endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
        self.assertEqual(endpoints["api_base"], "https://usdf-rsp-dev.slac.stanford.edu")
        # Check expected clients
        clients = ["consdb", "efd", "obsenv", "narrative_log", "exposure_log", "night_report"]
        endpoint_keys = list(endpoints.keys())
        self.assertTrue(len([c for c in clients if c not in endpoint_keys]) == 0)
