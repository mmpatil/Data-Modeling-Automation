'use strict';

module.exports = (sequelize, DataTypes) => {
  const ModelOutput = sequelize.define('ModelOutput', {
  	BGPVal: DataTypes.FLOAT,
	WhiteSkedacityPval: DataTypes.FLOAT,
	VIFPval: DataTypes.FLOAT,
	ADFResidual: DataTypes.FLOAT,
	RSquared: DataTypes.FLOAT,
	RMSE: DataTypes.FLOAT,
	MAE: DataTypes.FLOAT,
	MAPE: DataTypes.FLOAT,
	AIC: DataTypes.FLOAT,
	DynamicBacktestRange1MAPE: DataTypes.FLOAT,
	DynamicBacktestRange2MAPE: DataTypes.FLOAT,
	DynamicBacktestRange3MAPE: DataTypes.FLOAT,
	DynamicBacktestRange4MAPE: DataTypes.FLOAT,
	DynamicBacktestRange5MAPE: DataTypes.FLOAT,
	DynamicBacktestRange6MAPE: DataTypes.FLOAT,
	DynamicBacktestRange7MAPE: DataTypes.FLOAT,
	DynamicBacktestRange8MAPE: DataTypes.FLOAT,
	DynamicBacktestRange9MAPE: DataTypes.FLOAT,
	DynamicBacktestRange10MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange1MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange2MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange3MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange4MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange5MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange6MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange7MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange8MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange9MAPE: DataTypes.FLOAT,
	DynamicBacktestLongRange10MAPE: DataTypes.FLOAT,
	DurbinWatson1: DataTypes.FLOAT,
	DurbinWatson2: DataTypes.FLOAT,
	DurbinWatson3: DataTypes.FLOAT,
	DurbinWatson4: DataTypes.FLOAT,
	AcceptReject: DataTypes.BOOLEAN,
	AcceptRejectReason: DataTypes.TEXT,
	ShapiroWilk: DataTypes.FLOAT,
	BreuschPagan: DataTypes.FLOAT
  }, {
  	timestamps: false,
    freezeTableName: true,
    tableName: 'ModelOutput'
  });

  ModelOutput.associate = function(models) {
    models.ModelOutput.belongsTo(models.ModelRunDetail, {
      onDelete: "CASCADE",
      foreignKey: "ModelId"
    });
  };
  return ModelOutput;
};
