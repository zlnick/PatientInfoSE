/// 由IRIS提供，用于扩展AI能力，暴露IRIS上功能的API集合
/// Dispatch class defined by RESTSpec in MCPTools.spec
Class MCPTools.disp Extends %CSP.REST [ GeneratedBy = MCPTools.spec.cls, ProcedureBlock ]
{

/// The class containing the RESTSpec which generated this class
Parameter SpecificationClass = "MCPTools.spec";

/// Ignore any writes done directly by the REST method.
Parameter IgnoreWrites = 1;

/// Default the Content-Type for this application.
Parameter CONTENTTYPE = "application/json";

/// By default convert the input stream to Unicode
Parameter CONVERTINPUTSTREAM = 1;

XData UrlMap [ XMLNamespace = "http://www.intersystems.com/urlmap" ]
{
<Routes>
  <Route Url="/outpatient_appointment/:docId" Method="get" Call="getAppointments" />
</Routes>
}

ClassMethod getAppointments(pdocId As %String) As %Status
{
    Try {
        Do ##class(%REST.Impl).%SetContentType("application/json")
        If '##class(%REST.Impl).%CheckAccepts("application/json") {
            Try {
                Do ##class(MCPTools.impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            } Catch {
                Do ##class(%REST.Impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            }
            Quit
        }
        Set response=##class(MCPTools.impl).getAppointments(pdocId)
        Do ##class(MCPTools.impl).%WriteResponse(response)
    } Catch (ex) {
        Try {
            Do ##class(MCPTools.impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("MCPTools.impl","ExposeServerExceptions"))
        } Catch {
            Do ##class(%REST.Impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("MCPTools.impl","ExposeServerExceptions"))
        }
    }
    Quit $$$OK
}

}
