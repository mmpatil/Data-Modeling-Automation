'use strict';


module.exports = (sequelize, DataTypes) => {
  const IntermediateOutput = sequelize.define('IntermediateOutput', {
  	RegsDataframeJSON: DataTypes.TEXT,
    BaseDataframeJSON: DataTypes.TEXT,
    DependentJSON: DataTypes.TEXT
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'IntermediateOutput'
  });

  IntermediateOutput.associate = function(models) {
  	models.IntermediateOutput.belongsTo(models.RunDetail, {
      foreignKey:'RunId'
    });
  };
  return IntermediateOutput;
};
