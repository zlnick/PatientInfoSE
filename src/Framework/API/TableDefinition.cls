Class Framework.API.TableDefinition Extends %CSP.REST
{

/// Default the Content-Type for this application.
Parameter CONTENTTYPE = "application/json";

/// By default convert the input stream to Unicode
Parameter CONVERTINPUTSTREAM = 1;

XData UrlMap [ XMLNamespace = "http://www.intersystems.com/urlmap" ]
{
<Routes>
    <!-- 获得表定义 -->
    <Route Url="/TableDefinition/:namespace/:scheme" Method="get" Call="getTableDefs" />
    </Routes>
}

ClassMethod getTableDefs(namespace As %String, scheme As %String) As %Status
{
    Set tSC = $$$OK
    Try {
        Do ##class(%REST.Impl).%SetContentType("application/json")
        If '##class(%REST.Impl).%CheckAccepts("application/json") {
            Try {
                Do ##class(%REST.Impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            } Catch {
                Do ##class(%REST.Impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            }
            Quit
        }
        #Dim defs As %DynamicObject = {}
        Set defs=##class(Framework.TableUtils).GetTableDefByScheme(namespace,scheme)
        Do ##class(%REST.Impl).%WriteResponse(defs)
    } Catch (ex) {
        Try {
            Do ##class(%REST.Impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("%REST.Impl","ExposeServerExceptions"))
        } Catch {
            Do ##class(%REST.Impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("%REST.Impl","ExposeServerExceptions"))
        }
    }
    Quit $$$OK
}

}
