import http.client
import mimetypes

conn = http.client.HTTPSConnection("vkursi-api.azurewebsites.net")
payload = "{\"Edrpous\":[\"19124549\"],\"Ipns\":[\"3083707142\"],\"Koatuus\":[\"5621287500\"],\"Cadastrs\":[\"5621287500:03:001:0019\"],\"CalculateCost\":false,\"IsNeedUpdateAll\":false,\"IsReport\":true,\"TaskName\":\"Назва задачі\",\"DzkOnly\":false}"
headers = {
             'ContentType': 'application/json',
             'Authorization': 'Bearer eyJhbGciOiJIUzI1Ni...',
             'Content-Type': 'application/json',
             'Cookie': 'ARRAffinity=60c7763e47a70e864d73874a4687c10eb685afc08af8bda506303f7b37b172b8'
           }
conn.request("POST", "/api/1.0/estate/estatecreatetaskapi", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))