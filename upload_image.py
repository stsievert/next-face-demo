
import sys
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import os

TOKEN = os.environ.get("DRPBX_TOKEN", None)
if TOKEN is None:
    url = "https://blogs.dropbox.com/developers/2014/05/generate-an-access-token-for-your-own-account/"
    raise ValueError("DRPBX_TOKEN needs to be set to be one of Dropbox's"
                     "OAuth2 access tokens. For details, see " + url)

# make an instance of a Dropbox class which can make requests to the API.
dbx = dropbox.Dropbox(TOKEN)


def upload(fp, filename, verbose=True, app="WiSciFest"):
    """
    Assumes `filename` lives in the current directory and uploads to the
    Dropbox folder WiSciFest

    Returns the URL of the uploaded file
    """
    #  with open(filename, "rb") as f:
    with fp as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        # if '/' in filename:
        # filename = filename[filename.find('/') + 1:]
        # upload_path = filename.replace('/', '_')
        upload_path = filename
        if upload_path[0] not in {"/", "_"}:
            upload_path = "/" + upload_path
        if verbose:
            print("Uploading " + filename + " to Dropbox as " + upload_path + "...")
        # try:
        global dbx
        dbx = dropbox.Dropbox(TOKEN)
        dbx.files_upload(f.read(), upload_path, mode=WriteMode("overwrite"), mute=True)
        dropbox.sharing.CreateSharedLinkArg(upload_path)
        f = dbx.sharing_create_shared_link(upload_path)
        return f.url.replace("?dl=0", "?dl=1")


if __name__ == "__main__":
    # Check for an access token
    if len(TOKEN) == 0:
        sys.exit(
            "ERROR: Looks like you didn't add your access token. "
            "Open up backup-and-restore-example.py in a text editor and "
            "paste in your token in line 14."
        )

    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        sys.exit(
            "ERROR: Invalid access token; try re-generating an access "
            "token from the app console on the web."
        )

    # Create a backup of the current settings file
    url = upload("faces/11F_HA_C.png")
    print(url)
