Class MCP.API.MCPBridgeREST Extends %CSP.REST
{

// Parameter HandleCorsRequest = 0;

XData UrlMap [ XMLNamespace = "https://www.intersystems.com/urlmap" ]
{
<Routes>
    <Route Url="/add" Method="POST" Call="Add" />
    <Route Url="/get" Method="GET" Call="Get" />
</Routes>
}

ClassMethod Get() As %Status
{
    Set a = {}
    Set a.a = 1
    Set a.b = "3"
    D a.%ToJSON()
    Quit $$$OK
}

ClassMethod Add() As %Status
{
    Try {
        #Dim input As %DynamicObject = {}.%FromJSON(%request.Content)
        Set result = input.a + input.b
        Set response={}
        Set response.result = result
        Do ##class(%REST.Impl).%WriteResponse(response)
    } Catch (ex) {
        Do ##class(%REST.Impl).%SetStatusCode("400")
        return {"errormessage": "Client error"}
    }
    Quit $$$OK
}

}
