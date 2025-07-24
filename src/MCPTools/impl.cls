/// API for performing basic arithmetic operations<br/>
/// Business logic class defined by OpenAPI in MCPTools.spec<br/>
/// Created Jul 23, 2025 08:42:46
Class MCPTools.impl Extends %REST.Impl [ ProcedureBlock ]
{

/// If ExposeServerExceptions is true, then details of internal errors will be exposed.
Parameter ExposeServerExceptions = 0;

/// Returns the sum of two numbers<br/>
/// The method arguments hold values for:<br/>
///     numbers, The two numbers to add<br/>
ClassMethod addNumbers(numbers As %DynamicObject) As %DynamicObject
{
    Set rtnObj = {}
    Set rtnObj.result = numbers.a + numbers.b
    Quit rtnObj
}

}
