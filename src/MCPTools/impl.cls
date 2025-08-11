/// 由IRIS提供，用于扩展AI能力，暴露IRIS上功能的API集合<br/>
/// Business logic class defined by OpenAPI in MCPTools.spec<br/>
/// Updated Aug 7, 2025 08:23:39
Class MCPTools.impl Extends %REST.Impl [ ProcedureBlock ]
{

/// If ExposeServerExceptions is true, then details of internal errors will be exposed.
Parameter ExposeServerExceptions = 0;

/// 查询医生当天的门诊就诊预约。:param docId: 医生的FHIR资源id,:return: 查询结果（包含Appointment资源的FHIR Bundle JSON，其中每一条Appointment资源表示一次预约）<br/>
/// The method arguments hold values for:<br/>
///     docId, 医生的FHIR资源id<br/>
ClassMethod getAppointments(docId As %String) As %DynamicObject
{
    #Dim client As HS.FHIRServer.RestClient.HTTP = ##Class(HS.FHIRServer.RestClient.HTTP).CreateInstance("localFHIR")
    Do client.SetRequestFormat("JSON")
    Do client.SetResponseFormat("JSON")
    Set status = "booked"
    Set serviceType = "397"
    Set date = "eq"_$zd($h,3)
    Set queryString = "actor=Practitioner/"_docId_"&status="_status_"&service-type="_serviceType_"&date="_date
    #Dim fhirResp As HS.FHIRServer.API.Data.RestClient.Response = client.Search("GET","Appointment",,,queryString)
    #Dim result As %DynamicObject = {}.%FromJSON(fhirResp.Payload)
    quit result
    //(Place business logic here)
    //Do ..%SetStatusCode(<HTTP_status_code>)
    //Do ..%SetHeader(<name>,<value>)
    //Quit (Place response here) ; response may be a string, stream or dynamic object
}

}
