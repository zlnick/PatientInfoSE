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
  <!-- 两个数相加 -->
  <Route Url="/Math/add" Method="post" Call="addNumbers" />
  <Route Url="/outpatient_appointment/:docId" Method="get" Call="getAppointments" />
</Routes>
}

/// 两个数相加
ClassMethod addNumbers() As %Status
{
    Try {
        If $case(%request.ContentType,"application/json":0,:1) {
            Try {
                Do ##class(MCPTools.impl).%ReportRESTError(..#HTTP415UNSUPPORTEDMEDIATYPE,$$$ERROR($$$RESTContentType,%request.ContentType))
            } Catch {
                Do ##class(%REST.Impl).%ReportRESTError(..#HTTP415UNSUPPORTEDMEDIATYPE,$$$ERROR($$$RESTContentType,%request.ContentType))
            }
            Quit
        }
        Do ##class(%REST.Impl).%SetContentType("application/json")
        If '##class(%REST.Impl).%CheckAccepts("application/json") {
            Try {
                Do ##class(MCPTools.impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            } Catch {
                Do ##class(%REST.Impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts))
            }
            Quit
        }
        If '$isobject(%request.Content) || (%request.Content.Size=0) Do ##class(%REST.Impl).%ReportRESTError(..#HTTP400BADREQUEST,$$$ERROR($$$RESTRequired,"numbers")) Quit
        Set pnumbers=##class(%DynamicAbstractObject).%FromJSON(%request.Content)
        Set response=##class(MCPTools.impl).addNumbers(pnumbers)
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
