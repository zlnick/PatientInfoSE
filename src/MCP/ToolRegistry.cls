Class MCP.ToolRegistry Extends %Persistent
{

Property Name As %String;

Property ClassName As %String;

Property Description As %String(MAXLEN = 1024);

Property InputSchema As %DynamicObject;

Property OutputSchema As %DynamicObject;

// "tool" 或 "agent"

Property Type As %String;

Property Enabled As %Boolean;

Index ToolNameIndex On Name [ Unique ];

Storage Default
{
<Data name="InputSchema">
<Attribute>InputSchema</Attribute>
<Structure>node</Structure>
<Subscript>"InputSchema"</Subscript>
</Data>
<Data name="OutputSchema">
<Attribute>OutputSchema</Attribute>
<Structure>node</Structure>
<Subscript>"OutputSchema"</Subscript>
</Data>
<Data name="ToolRegistryDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>Name</Value>
</Value>
<Value name="3">
<Value>ClassName</Value>
</Value>
<Value name="4">
<Value>Description</Value>
</Value>
<Value name="5">
<Value>Type</Value>
</Value>
<Value name="6">
<Value>Enabled</Value>
</Value>
</Data>
<DataLocation>^MCP.ToolRegistryD</DataLocation>
<DefaultData>ToolRegistryDefaultData</DefaultData>
<IdLocation>^MCP.ToolRegistryD</IdLocation>
<IndexLocation>^MCP.ToolRegistryI</IndexLocation>
<StreamLocation>^MCP.ToolRegistryS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
