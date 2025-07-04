import argparse
import logging

from astropy.time import Time, TimeDelta

from rubin_nights import connections, scriptqueue, scriptqueue_formatting

logging.getLogger(__name__).setLevel(logging.INFO)


def night_story(day_obs: str | None = None, tokenfile: str | None = None, site: str | None = None) -> str:
    """Generate HTML containing the narrative log and
     addBlock/Scheduler enable tasks, as well as any obs_env updates.

     Parameters
     ----------
     day_obs : `str` or None
        The day_obs in format YYYY-MM-DD.
        The resulting HTML will cover the entire day_obs.
        Default None uses today's day_obs.
    tokenfile : `str` or None
        Passed to `get_access_token`.
        Default of None is perfect for any RSP.
        Running outside of an RSP requires an RSP token, see more at
        https://nb.lsst.io/environment/tokens.html
    site : `str` or None
        Override site location to a preferred site.
        Should match source of tokenfile, if specified.

    Returns
    -------
    html : `str`
        The HTML formatted time, block or Scheduler configuration,
         a short description (or narrative log message) and reporter.
    """
    if day_obs is None:
        day_obs = Time(Time.now().mjd - 0.5, format="mjd", scale="tai").iso[0:10]

    day_obs_time = Time(f"{day_obs}T12:00:00", format="isot", scale="tai")

    t_start = day_obs_time
    t_end = day_obs_time + TimeDelta(1, format="jd")

    endpoints = connections.get_clients(tokenfile=tokenfile, site=site)
    efd_and_messages, cols = scriptqueue.get_consolidated_messages(t_start, t_end, endpoints)

    print(f"Retrieved a total of {len(efd_and_messages)} in consolidated messages for {day_obs}")

    # Only Simonyi messages:
    only_simonyi = efd_and_messages.query(
        "finalStatus == 'Job Change' | "
        "(salIndex == 0 and not name.str.contains('Aux')) | "
        "(name == 'Obsenv Update')"
    )
    html = scriptqueue_formatting.format_html(only_simonyi, cols=cols, time_order="oldest")
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day_obs", type=str, default=None)
    parser.add_argument("--tokenfile", type=str, default=None)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    html = night_story(day_obs=args.day_obs, tokenfile=args.tokenfile, site=args.site)

    if args.outfile is None:
        args.outfile = "night_story.html"
    with open(args.outfile, "w") as f:
        f.writelines(html)
