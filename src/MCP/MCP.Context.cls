Class MCP.Context Extends %Persistent
{

Property SessionId As %String;

Property PatientId As %String;

Property InputText As %String;

Property TaskChain As %DynamicObject;

Property CreatedAt As %TimeStamp;

Property UpdatedAt As %TimeStamp;

Storage Default
{
<Data name="ContextDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>SessionId</Value>
</Value>
<Value name="3">
<Value>PatientId</Value>
</Value>
<Value name="4">
<Value>InputText</Value>
</Value>
<Value name="5">
<Value>CreatedAt</Value>
</Value>
<Value name="6">
<Value>UpdatedAt</Value>
</Value>
</Data>
<Data name="TaskChain">
<Attribute>TaskChain</Attribute>
<Structure>node</Structure>
<Subscript>"TaskChain"</Subscript>
</Data>
<DataLocation>^MCP.ContextD</DataLocation>
<DefaultData>ContextDefaultData</DefaultData>
<IdLocation>^MCP.ContextD</IdLocation>
<IndexLocation>^MCP.ContextI</IndexLocation>
<StreamLocation>^MCP.ContextS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
