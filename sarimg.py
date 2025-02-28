import numpy
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import io
import cv2
import base64

# Your client credentials
client_id = 'sh-f1d52186-5cb8-4f23-9599-dbe672603dc9'
client_secret = 'd0YjlYCRdiqfBcCF8pXPwpJakZgTocxg'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)
'''evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["VV"],
    output: { id: "default", bands: 1 },
  }
}

function evaluatePixel(samples) {
  return [2 * samples.VV]
}
"""

request = {
    "input": {
        "bounds": {
            "bbox": [
                1360000,
                5121900,
                1370000,
                5131900,
            ],
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
        },
        "data": [
            {
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": "2019-02-02T00:00:00Z",
                        "to": "2019-04-02T23:59:59Z",
                    }
                },
                "processing": {"orthorectify": "true"},
            }
        ],
    },
    "output": {
        "width": 512,
        "height": 512,
        "responses": [
            {
                "identifier": "default",
                "format": {"type": "image/png"},
            }
        ],
    },
    "evalscript": evalscript,
}

url = "https://sh.dataspace.copernicus.eu/api/v1/process"
response = oauth.post(url, json=request)'''

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: {
      bands: 3,
      sampleType: "AUTO", // default value - scales the output values from [0,1] to [0,255].
    },
  }
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
}
"""

request = {
    "input": {
        "bounds": {
            "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            "bbox": [48.325487506118264, 28.414218984218607, 48.34387693420447, 28.442803969343313],
        },
        "data": [
            {
                "type": "sentinel-2-l1c",
                "dataFilter": {
                    "timeRange": {
                        "from": "2017-08-09T00:00:00Z",
                        "to": "2017-08-12T00:00:00Z",
                    }
                },
            }
        ],
    },
    "output": {
        "width": 512,
        "height": 512,
    },
    "evalscript": evalscript,
}

url = "https://sh.dataspace.copernicus.eu/api/v1/process"
response = oauth.post(url, json=request)
image_bytes = io.BytesIO(response.content)
img = Image.open(image_bytes)
img.save('my-image.jpeg')

# flatNumpyArray = numpy.array(image_bytes)
flatNumpyArrayL = numpy.array(img.convert('L'))
flatNumpyArrayRGB = numpy.array(img.convert('RGB'))# Convert image to grayscale and then to a numpy array
# Convert the array to make a 400x300 grayscale image.
grayImage = flatNumpyArrayL.reshape(512, 512)
cv2.imwrite('RandomGray.png', grayImage)

# Convert the array to make a 400x100 color image.
bgrImage = flatNumpyArrayRGB.reshape(512, 512, 3)
cv2.imwrite('RandomColor.png', bgrImage)

# All requests using this session will have an access token automatically added
resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
print(resp.content)