Class Framework.TableUtils Extends %RegisteredObject
{

ClassMethod GetTableDefByScheme(Namespace As %String, SchemName As %String) As %DynamicObject
{
    Set ns = ##Class(%SYSTEM.SYS).NameSpace()
    #Dim tableList As %DynamicArray = []
    zn Namespace
    Set tabName = ""
    Set tblDesc = ""
    //表定义
   &sql(DECLARE tblCursor CURSOR FOR 
        SELECT TABLE_NAME, DESCRIPTION
        INTO :tabName,:tblDesc
        FROM INFORMATION_SCHEMA.TABLES
        where TABLE_SCHEMA = :SchemName
    )
   &sql(OPEN tblCursor)
      QUIT:(SQLCODE'=0)
   FOR { &sql(FETCH tblCursor)
        QUIT:SQLCODE  
        //WRITE !," TableName=",tabName," Desc=",tblDesc
        if ("" '= tabName){
            #Dim tableDef As %DynamicObject = {}
            Set tableDef.TableName = SchemName_"."_tabName
            Set tableDef.Description = tblDesc
            Set tableDef.Columns = []
            SET colName="",type="",colDesc=""
            //列定义
            &sql(DECLARE colCursor CURSOR FOR 
                SELECT COLUMN_NAME, DATA_TYPE, DESCRIPTION
                INTO :colName,:type,:colDesc
                FROM INFORMATION_SCHEMA.COLUMNS
                where TABLE_SCHEMA = :SchemName
                and TABLE_NAME = :tabName)
            &sql(OPEN colCursor)
                QUIT:(SQLCODE'=0)
            FOR { &sql(FETCH colCursor)
                QUIT:SQLCODE  
                //WRITE !," ColumnName=",colName," Type=",type," Desc=",colDesc
                /*#Dim colDef As Framework.Data.ColumnDefinition = ##Class(Framework.Data.ColumnDefinition).%New()
                Set colDef.ColumnName = colName
                Set colDef.Type = type
                Set colDef.Description = colDesc
                Do tableDef.Columns.Insert(colDef)*/
                #Dim colDef As %DynamicObject = {}
                Set colDef.ColumnName = colName
                Set colDef.Type = type
                Set colDef.Description = colDesc
                Do tableDef.Columns.%Push(colDef)
            }
            &sql(CLOSE colCursor)
            Do tableList.%Push(tableDef)
        }
 }
   &sql(CLOSE tblCursor)
   zn ns
   Quit tableList
}

ClassMethod TestRun()
{
    Set ns = "MCP"
    Set schema = "Data"
    Set tables = ..GetTableDefByScheme(ns,schema)
    zw tables
}

}
